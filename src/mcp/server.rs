//! MCP Server Mode - JSON-RPC 2.0 over stdio for Model Context Protocol.
//!
//! KOS P6: Exposes Nagual's tools (search, store, coherence, lineage, etc.)
//! as MCP-compatible tool endpoints via a lightweight JSON-RPC 2.0 protocol handler.
//!
//! # Architecture
//!
//! ```text
//! stdin (JSON-RPC) --> McpServer::parse_request()
//!                          |
//!                          v
//!                  McpServer::handle_request()
//!                          |
//!                    +-----+-----+-----+
//!                    |           |     |
//!                    v           v     v
//!             initialize   tools/list  tools/call
//!                                      |
//!                                      v
//!                               handle_tool_call()
//!                                      |
//!                                      v
//!                        stdout (JSON-RPC response)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::mcp::server::{McpServer, McpServerConfig};
//!
//! let server = McpServer::new(McpServerConfig::default());
//!
//! let request = McpServer::parse_request(
//!     r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#
//! ).unwrap();
//!
//! let response = server.handle_request(&request);
//! println!("{}", serde_json::to_string(&response).unwrap());
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{McpRegistry, NagualContext};

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version - always "2.0".
    pub jsonrpc: String,
    /// Request identifier (number or string). `None` for notifications.
    #[serde(default)]
    pub id: Option<Value>,
    /// The method to invoke.
    pub method: String,
    /// Method parameters.
    #[serde(default)]
    pub params: Option<Value>,
}

/// A JSON-RPC 2.0 response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// Protocol version - always "2.0".
    pub jsonrpc: String,
    /// Request identifier echoed back from the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    /// Result on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Numeric error code.
    pub code: i64,
    /// Human-readable error message.
    pub message: String,
    /// Optional structured error data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    /// Standard parse error (-32700).
    pub const PARSE_ERROR: i64 = -32700;
    /// Standard invalid request (-32600).
    pub const INVALID_REQUEST: i64 = -32600;
    /// Standard method not found (-32601).
    pub const METHOD_NOT_FOUND: i64 = -32601;
    /// Standard invalid params (-32602).
    pub const INVALID_PARAMS: i64 = -32602;
    /// Standard internal error (-32603).
    pub const INTERNAL_ERROR: i64 = -32603;

    /// Create a parse error.
    pub fn parse_error(detail: &str) -> Self {
        Self {
            code: Self::PARSE_ERROR,
            message: format!("Parse error: {detail}"),
            data: None,
        }
    }

    /// Create an invalid request error.
    pub fn invalid_request(detail: &str) -> Self {
        Self {
            code: Self::INVALID_REQUEST,
            message: format!("Invalid request: {detail}"),
            data: None,
        }
    }

    /// Create a method not found error.
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: Self::METHOD_NOT_FOUND,
            message: format!("Method not found: {method}"),
            data: None,
        }
    }

    /// Create an invalid params error.
    pub fn invalid_params(detail: &str) -> Self {
        Self {
            code: Self::INVALID_PARAMS,
            message: format!("Invalid params: {detail}"),
            data: None,
        }
    }

    /// Create an internal error.
    pub fn internal_error(detail: &str) -> Self {
        Self {
            code: Self::INTERNAL_ERROR,
            message: format!("Internal error: {detail}"),
            data: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MCP protocol types
// ---------------------------------------------------------------------------

/// Describes a single MCP tool with its JSON Schema input specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    /// Tool name (e.g., "nagual_search").
    pub name: String,
    /// Human-readable description of the tool.
    pub description: String,
    /// JSON Schema describing accepted input parameters.
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Configuration for the MCP server.
pub struct McpServerConfig {
    /// Server name advertised during initialization.
    pub server_name: String,
    /// Server version advertised during initialization.
    pub server_version: String,
    /// Maximum allowed request size in bytes.
    pub max_request_size: usize,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            server_name: "nagual".to_string(),
            server_version: "1.1.0".to_string(),
            max_request_size: 1_000_000,
            timeout_ms: 30_000,
        }
    }
}

/// MCP capabilities advertised by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// Whether the server exposes tools.
    pub tools: bool,
    /// Whether the server exposes resources.
    pub resources: bool,
    /// Whether the server exposes prompts.
    pub prompts: bool,
}

impl Default for McpCapabilities {
    fn default() -> Self {
        Self {
            tools: true,
            resources: false,
            prompts: false,
        }
    }
}

// ---------------------------------------------------------------------------
// McpServer - lightweight protocol handler
// ---------------------------------------------------------------------------

/// Lightweight MCP protocol handler.
///
/// Parses JSON-RPC 2.0 requests, routes them to the appropriate handler, and
/// produces JSON-RPC 2.0 responses. This struct does **not** perform I/O; the
/// caller is responsible for reading from stdin / writing to stdout.
pub struct McpServer {
    /// Server configuration.
    config: McpServerConfig,
    /// Registered tools.
    tools: Vec<McpToolInfo>,
    /// Capabilities advertised during initialization.
    capabilities: McpCapabilities,
    /// Optional MCP registry for actual tool execution.
    registry: Option<Arc<McpRegistry>>,
    /// Optional Nagual context for tool execution.
    context: Option<Arc<NagualContext>>,
}

impl McpServer {
    /// Create a new MCP server with the given configuration.
    ///
    /// The 7 default Nagual tools are registered automatically.
    pub fn new(config: McpServerConfig) -> Self {
        let tools = Self::default_tools();
        Self {
            config,
            tools,
            capabilities: McpCapabilities::default(),
            registry: None,
            context: None,
        }
    }

    /// Attach an MCP registry and context for live tool execution.
    ///
    /// When a registry is attached, `handle_tool_call` delegates to
    /// `McpRegistry::execute()` instead of returning a placeholder.
    pub fn with_registry(
        mut self,
        registry: Arc<McpRegistry>,
        context: Arc<NagualContext>,
    ) -> Self {
        self.registry = Some(registry);
        self.context = Some(context);
        self
    }

    /// Replace the tool list with a custom set.
    pub fn with_tools(mut self, tools: Vec<McpToolInfo>) -> Self {
        self.tools = tools;
        self
    }

    /// Register a single additional tool.
    pub fn register_tool(&mut self, tool: McpToolInfo) {
        // Avoid duplicates by name.
        if !self.tools.iter().any(|t| t.name == tool.name) {
            self.tools.push(tool);
        }
    }

    /// Return a reference to the registered tools.
    pub fn tools(&self) -> &[McpToolInfo] {
        &self.tools
    }

    /// Return a reference to the capabilities.
    pub fn capabilities(&self) -> &McpCapabilities {
        &self.capabilities
    }

    // -----------------------------------------------------------------------
    // Request parsing
    // -----------------------------------------------------------------------

    /// Parse a raw JSON string into a [`JsonRpcRequest`].
    ///
    /// Returns a [`JsonRpcError`] on parse failure.
    pub fn parse_request(input: &str) -> Result<JsonRpcRequest, JsonRpcError> {
        if input.trim().is_empty() {
            return Err(JsonRpcError::parse_error("empty input"));
        }

        let value: Value = serde_json::from_str(input)
            .map_err(|e| JsonRpcError::parse_error(&e.to_string()))?;

        // Reject batch requests (arrays).
        if value.is_array() {
            return Err(JsonRpcError::invalid_request("batch requests are not supported"));
        }

        let request: JsonRpcRequest = serde_json::from_value(value)
            .map_err(|e| JsonRpcError::invalid_request(&e.to_string()))?;

        // Validate jsonrpc field.
        if request.jsonrpc != "2.0" {
            return Err(JsonRpcError::invalid_request(
                "jsonrpc field must be \"2.0\"",
            ));
        }

        Ok(request)
    }

    // -----------------------------------------------------------------------
    // Request routing
    // -----------------------------------------------------------------------

    /// Route a parsed request to the appropriate handler and return a response.
    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(request.id.clone()),
            "tools/list" => self.handle_tools_list(request.id.clone()),
            "tools/call" => {
                let params = request.params.as_ref().unwrap_or(&Value::Null);
                self.handle_tool_call(request.id.clone(), params)
            }
            "notifications/initialized" => {
                // Notifications have no response, but we return a no-op envelope.
                JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: None,
                }
            }
            _ => Self::make_error_response(
                request.id.clone(),
                JsonRpcError::method_not_found(&request.method),
            ),
        }
    }

    // -----------------------------------------------------------------------
    // MCP handlers
    // -----------------------------------------------------------------------

    /// Handle the MCP `initialize` request.
    ///
    /// Returns server info and advertised capabilities.
    pub fn handle_initialize(&self, id: Option<Value>) -> JsonRpcResponse {
        Self::make_success_response(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.config.server_name,
                    "version": self.config.server_version,
                },
                "capabilities": {
                    "tools": self.capabilities.tools,
                    "resources": self.capabilities.resources,
                    "prompts": self.capabilities.prompts,
                },
            }),
        )
    }

    /// Handle the MCP `tools/list` request.
    ///
    /// Returns all registered tool definitions.
    pub fn handle_tools_list(&self, id: Option<Value>) -> JsonRpcResponse {
        let tools_json: Vec<Value> = self
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();

        Self::make_success_response(id, json!({ "tools": tools_json }))
    }

    /// Handle the MCP `tools/call` request.
    ///
    /// Validates that the requested tool exists and that the required
    /// `"name"` field is present in `params`. The actual execution is not
    /// performed here -- that requires a [`NagualContext`] wired up by the
    /// caller. Instead, a placeholder acknowledgment is returned.
    pub fn handle_tool_call(&self, id: Option<Value>, params: &Value) -> JsonRpcResponse {
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                return Self::make_error_response(
                    id,
                    JsonRpcError::invalid_params("missing required field: \"name\""),
                );
            }
        };

        // Check that the tool exists.
        let tool = self.tools.iter().find(|t| t.name == tool_name);
        if tool.is_none() {
            return Self::make_error_response(
                id,
                JsonRpcError::method_not_found(&format!("unknown tool: {tool_name}")),
            );
        }

        let arguments = params.get("arguments").cloned().unwrap_or(Value::Object(
            serde_json::Map::new(),
        ));

        // Delegate to registry if available, otherwise return tool info.
        if let (Some(registry), Some(context)) = (&self.registry, &self.context) {
            // Use tokio::task::block_in_place for sync->async bridge
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(registry.execute(tool_name, &arguments, context))
            });
            match result {
                Ok(tool_result) => Self::make_success_response(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string(&tool_result.output)
                                .unwrap_or_else(|_| tool_result.output.to_string()),
                        }],
                        "isError": false,
                    }),
                ),
                Err(e) => Self::make_success_response(
                    id,
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Tool execution error: {}", e),
                        }],
                        "isError": true,
                    }),
                ),
            }
        } else {
            // No registry attached -- return informational response.
            Self::make_success_response(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": format!(
                            "Tool '{}' registered. Attach a McpRegistry via with_registry() for execution.",
                            tool_name,
                        ),
                    }],
                    "tool": tool_name,
                    "arguments": arguments,
                    "isError": false,
                }),
            )
        }
    }

    // -----------------------------------------------------------------------
    // Response helpers
    // -----------------------------------------------------------------------

    /// Construct an error response.
    pub fn make_error_response(id: Option<Value>, error: JsonRpcError) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Construct a success response.
    pub fn make_success_response(id: Option<Value>, result: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    // -----------------------------------------------------------------------
    // Default tools
    // -----------------------------------------------------------------------

    /// Returns the 7 standard Nagual MCP tools with their JSON Schema definitions.
    pub fn default_tools() -> Vec<McpToolInfo> {
        vec![
            McpToolInfo {
                name: "nagual_search".to_string(),
                description: "Search patterns in the Nagual knowledge base by text query. \
                    Returns matching patterns ranked by relevance."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (natural language or keywords)"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "Maximum number of results"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Filter by domain (e.g., 'rust.async')"
                        },
                        "min_reward": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum reward threshold"
                        }
                    },
                    "required": ["query"]
                }),
            },
            McpToolInfo {
                name: "nagual_store".to_string(),
                description: "Store a new reasoning pattern in the knowledge base. \
                    Captures a problem-solution pair with domain and tags."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "Problem or challenge description"
                        },
                        "solution": {
                            "type": "string",
                            "description": "Solution or approach"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain using dot notation (e.g., 'rust.async')"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for categorization"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Initial confidence (0.0-1.0)"
                        }
                    },
                    "required": ["problem", "solution"]
                }),
            },
            McpToolInfo {
                name: "nagual_get".to_string(),
                description: "Retrieve a single pattern by its ID."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pattern_id": {
                            "type": "string",
                            "description": "The pattern identifier"
                        }
                    },
                    "required": ["pattern_id"]
                }),
            },
            McpToolInfo {
                name: "nagual_coherence".to_string(),
                description: "Check coherence between two or more patterns. \
                    Returns a similarity/conflict score and explanation."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pattern_ids": {
                            "type": "array",
                            "items": { "type": "string" },
                            "minItems": 2,
                            "description": "Pattern IDs to compare"
                        },
                        "threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "Coherence threshold for flagging conflicts"
                        }
                    },
                    "required": ["pattern_ids"]
                }),
            },
            McpToolInfo {
                name: "nagual_lineage".to_string(),
                description: "Get the lineage (derivation tree) of a pattern. \
                    Shows parent patterns, children, and evolution history."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "pattern_id": {
                            "type": "string",
                            "description": "Root pattern ID for the lineage query"
                        },
                        "depth": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 3,
                            "description": "Maximum depth of the lineage tree"
                        }
                    },
                    "required": ["pattern_id"]
                }),
            },
            McpToolInfo {
                name: "nagual_domains".to_string(),
                description: "List all domains in the knowledge base with pattern counts."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "min_patterns": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Only return domains with at least this many patterns"
                        }
                    }
                }),
            },
            McpToolInfo {
                name: "nagual_recommend".to_string(),
                description: "Get learning recommendations based on pattern performance. \
                    Identifies weak areas and suggests improvements."
                    .to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Domain to get recommendations for"
                        },
                        "max_recommendations": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "default": 5,
                            "description": "Maximum number of recommendations"
                        }
                    }
                }),
            },
        ]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // JsonRpcRequest parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_valid_request() {
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = McpServer::parse_request(input).unwrap();
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Value::from(1)));
    }

    #[test]
    fn test_parse_request_missing_method() {
        let input = r#"{"jsonrpc":"2.0","id":1}"#;
        let err = McpServer::parse_request(input).unwrap_err();
        assert_eq!(err.code, JsonRpcError::INVALID_REQUEST);
    }

    #[test]
    fn test_parse_request_invalid_json() {
        let err = McpServer::parse_request("not json at all").unwrap_err();
        assert_eq!(err.code, JsonRpcError::PARSE_ERROR);
    }

    #[test]
    fn test_parse_request_batch_not_supported() {
        let input = r#"[{"jsonrpc":"2.0","id":1,"method":"a"},{"jsonrpc":"2.0","id":2,"method":"b"}]"#;
        let err = McpServer::parse_request(input).unwrap_err();
        assert_eq!(err.code, JsonRpcError::INVALID_REQUEST);
        assert!(err.message.contains("batch"));
    }

    #[test]
    fn test_parse_request_empty_string() {
        let err = McpServer::parse_request("").unwrap_err();
        assert_eq!(err.code, JsonRpcError::PARSE_ERROR);
        assert!(err.message.contains("empty"));
    }

    // -----------------------------------------------------------------------
    // JsonRpcResponse construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_success_response() {
        let resp = McpServer::make_success_response(Some(Value::from(42)), json!({"ok": true}));
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, Some(Value::from(42)));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_error_response() {
        let resp = McpServer::make_error_response(
            Some(Value::from(1)),
            JsonRpcError::internal_error("boom"),
        );
        assert_eq!(resp.jsonrpc, "2.0");
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, JsonRpcError::INTERNAL_ERROR);
    }

    #[test]
    fn test_response_with_null_id() {
        let resp = McpServer::make_success_response(None, json!("ok"));
        assert!(resp.id.is_none());
        assert_eq!(resp.result, Some(json!("ok")));
    }

    // -----------------------------------------------------------------------
    // JsonRpcError tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_error_code() {
        let err = JsonRpcError::parse_error("bad json");
        assert_eq!(err.code, -32700);
        assert!(err.message.contains("bad json"));
    }

    #[test]
    fn test_invalid_request_code() {
        let err = JsonRpcError::invalid_request("missing field");
        assert_eq!(err.code, -32600);
    }

    #[test]
    fn test_method_not_found_code() {
        let err = JsonRpcError::method_not_found("foo/bar");
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("foo/bar"));
    }

    #[test]
    fn test_internal_error_code() {
        let err = JsonRpcError::internal_error("kaboom");
        assert_eq!(err.code, -32603);
    }

    // -----------------------------------------------------------------------
    // McpServerConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = McpServerConfig::default();
        assert_eq!(cfg.server_name, "nagual");
        assert_eq!(cfg.server_version, "1.1.0");
        assert_eq!(cfg.max_request_size, 1_000_000);
        assert_eq!(cfg.timeout_ms, 30_000);
    }

    #[test]
    fn test_config_custom() {
        let cfg = McpServerConfig {
            server_name: "custom".to_string(),
            server_version: "2.0.0".to_string(),
            max_request_size: 500_000,
            timeout_ms: 10_000,
        };
        assert_eq!(cfg.server_name, "custom");
        assert_eq!(cfg.server_version, "2.0.0");
        assert_eq!(cfg.max_request_size, 500_000);
        assert_eq!(cfg.timeout_ms, 10_000);
    }

    // -----------------------------------------------------------------------
    // McpCapabilities tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_capabilities_defaults() {
        let caps = McpCapabilities::default();
        assert!(caps.tools);
        assert!(!caps.resources);
        assert!(!caps.prompts);
    }

    // -----------------------------------------------------------------------
    // McpServer::new tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_server_has_default_tools() {
        let server = McpServer::new(McpServerConfig::default());
        assert!(!server.tools().is_empty());
        let names: Vec<&str> = server.tools().iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"nagual_search"));
        assert!(names.contains(&"nagual_store"));
    }

    #[test]
    fn test_server_default_tool_count() {
        let server = McpServer::new(McpServerConfig::default());
        assert_eq!(server.tools().len(), 7);
    }

    // -----------------------------------------------------------------------
    // register_tool tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_tool_adds_tool() {
        let mut server = McpServer::new(McpServerConfig::default());
        let initial = server.tools().len();
        server.register_tool(McpToolInfo {
            name: "custom_tool".to_string(),
            description: "A custom tool".to_string(),
            input_schema: json!({"type": "object"}),
        });
        assert_eq!(server.tools().len(), initial + 1);
        assert!(server.tools().iter().any(|t| t.name == "custom_tool"));
    }

    #[test]
    fn test_register_tool_no_duplicate() {
        let mut server = McpServer::new(McpServerConfig::default());
        let initial = server.tools().len();
        // Try to register a tool with a name that already exists.
        server.register_tool(McpToolInfo {
            name: "nagual_search".to_string(),
            description: "duplicate".to_string(),
            input_schema: json!({"type": "object"}),
        });
        assert_eq!(server.tools().len(), initial);
    }

    // -----------------------------------------------------------------------
    // handle_initialize tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_handle_initialize_returns_server_info() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_initialize(Some(Value::from(1)));
        let result = resp.result.unwrap();
        let server_info = result.get("serverInfo").unwrap();
        assert_eq!(server_info.get("name").unwrap(), "nagual");
    }

    #[test]
    fn test_handle_initialize_returns_capabilities() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_initialize(Some(Value::from(1)));
        let result = resp.result.unwrap();
        let caps = result.get("capabilities").unwrap();
        assert_eq!(caps.get("tools").unwrap(), &Value::Bool(true));
        assert_eq!(caps.get("resources").unwrap(), &Value::Bool(false));
    }

    #[test]
    fn test_handle_initialize_correct_version() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_initialize(Some(Value::from(1)));
        let result = resp.result.unwrap();
        let version = result
            .get("serverInfo")
            .unwrap()
            .get("version")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(version, "1.1.0");
    }

    // -----------------------------------------------------------------------
    // handle_tools_list tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_handle_tools_list_returns_all_tools() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_tools_list(Some(Value::from(2)));
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 7);
    }

    #[test]
    fn test_handle_tools_list_each_has_name_description_schema() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_tools_list(Some(Value::from(2)));
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        for tool in tools {
            assert!(tool.get("name").is_some(), "tool missing name");
            assert!(tool.get("description").is_some(), "tool missing description");
            assert!(tool.get("inputSchema").is_some(), "tool missing inputSchema");
        }
    }

    #[test]
    fn test_handle_tools_list_correct_tool_count() {
        let server = McpServer::new(McpServerConfig::default());
        let resp = server.handle_tools_list(Some(Value::from(2)));
        let result = resp.result.unwrap();
        let tools = result.get("tools").unwrap().as_array().unwrap();
        // Verify specific tool names.
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t.get("name").unwrap().as_str().unwrap())
            .collect();
        assert!(names.contains(&"nagual_search"));
        assert!(names.contains(&"nagual_store"));
        assert!(names.contains(&"nagual_get"));
        assert!(names.contains(&"nagual_coherence"));
        assert!(names.contains(&"nagual_lineage"));
        assert!(names.contains(&"nagual_domains"));
        assert!(names.contains(&"nagual_recommend"));
    }

    // -----------------------------------------------------------------------
    // handle_tool_call tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_handle_tool_call_valid_tool() {
        let server = McpServer::new(McpServerConfig::default());
        let params = json!({"name": "nagual_search", "arguments": {"query": "rust async"}});
        let resp = server.handle_tool_call(Some(Value::from(3)), &params);
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result.get("tool").unwrap(), "nagual_search");
        assert_eq!(result.get("isError").unwrap(), &Value::Bool(false));
    }

    #[test]
    fn test_handle_tool_call_unknown_tool() {
        let server = McpServer::new(McpServerConfig::default());
        let params = json!({"name": "nonexistent_tool"});
        let resp = server.handle_tool_call(Some(Value::from(3)), &params);
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            JsonRpcError::METHOD_NOT_FOUND,
        );
    }

    #[test]
    fn test_handle_tool_call_missing_name() {
        let server = McpServer::new(McpServerConfig::default());
        let params = json!({"arguments": {"query": "test"}});
        let resp = server.handle_tool_call(Some(Value::from(3)), &params);
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            JsonRpcError::INVALID_PARAMS,
        );
    }

    #[test]
    fn test_handle_tool_call_missing_arguments() {
        let server = McpServer::new(McpServerConfig::default());
        let params = json!({"name": "nagual_search"});
        let resp = server.handle_tool_call(Some(Value::from(3)), &params);
        // Should still succeed -- missing arguments default to empty object.
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result.get("arguments").unwrap(), &json!({}));
    }

    // -----------------------------------------------------------------------
    // handle_request routing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_route_initialize() {
        let server = McpServer::new(McpServerConfig::default());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(1)),
            method: "initialize".to_string(),
            params: Some(json!({"clientInfo": {"name": "test", "version": "1.0"}})),
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        assert!(resp.result.as_ref().unwrap().get("serverInfo").is_some());
    }

    #[test]
    fn test_route_tools_list() {
        let server = McpServer::new(McpServerConfig::default());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(2)),
            method: "tools/list".to_string(),
            params: None,
        };
        let resp = server.handle_request(&req);
        assert!(resp.result.is_some());
        assert!(resp.result.as_ref().unwrap().get("tools").is_some());
    }

    #[test]
    fn test_route_tools_call() {
        let server = McpServer::new(McpServerConfig::default());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(3)),
            method: "tools/call".to_string(),
            params: Some(json!({"name": "nagual_domains"})),
        };
        let resp = server.handle_request(&req);
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_route_unknown_method() {
        let server = McpServer::new(McpServerConfig::default());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::from(99)),
            method: "unknown/method".to_string(),
            params: None,
        };
        let resp = server.handle_request(&req);
        assert!(resp.error.is_some());
        assert_eq!(
            resp.error.as_ref().unwrap().code,
            JsonRpcError::METHOD_NOT_FOUND,
        );
    }

    // -----------------------------------------------------------------------
    // McpToolInfo tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_info_schema_structure() {
        let tool = McpToolInfo {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }),
        };
        assert!(tool.input_schema.is_object());
        assert_eq!(
            tool.input_schema.get("type").unwrap().as_str().unwrap(),
            "object",
        );
        assert!(tool.input_schema.get("properties").is_some());
    }

    #[test]
    fn test_all_default_tools_have_schemas() {
        let tools = McpServer::default_tools();
        for tool in &tools {
            assert!(
                tool.input_schema.is_object(),
                "Tool '{}' has non-object schema",
                tool.name,
            );
            assert!(
                tool.input_schema.get("type").is_some(),
                "Tool '{}' schema missing 'type'",
                tool.name,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Notification handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_notifications_initialized_empty_response() {
        let server = McpServer::new(McpServerConfig::default());
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let resp = server.handle_request(&req);
        assert!(resp.id.is_none());
        assert!(resp.result.is_none());
        assert!(resp.error.is_none());
    }
}
