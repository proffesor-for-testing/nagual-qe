//! `nagual activity status` — health check + device listing + DB stats.

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::resolve_screenpipe_url;
use super::types::*;
use super::StatusArgs;

pub(super) async fn run_status(args: &StatusArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    match client.health().await {
        Some(health) => {
            let monitors = client.vision_monitors().await;
            let audio_devices = client.audio_devices().await;

            let db_rows = match client
                .raw_sql(
                    "SELECT \
                     (SELECT COUNT(*) FROM frames) as frames, \
                     (SELECT COUNT(*) FROM ocr_text) as ocr, \
                     (SELECT COUNT(*) FROM audio_transcriptions) as audio, \
                     (SELECT COUNT(*) FROM ui_monitoring) as ui_monitoring, \
                     (SELECT COUNT(*) FROM accessibility) as accessibility",
                )
                .await
            {
                Ok(rows) if !rows.is_empty() => {
                    let r = &rows[0];
                    DbRowCounts {
                        frames: r["frames"].as_u64().unwrap_or(0) as usize,
                        ocr: r["ocr"].as_u64().unwrap_or(0) as usize,
                        audio: r["audio"].as_u64().unwrap_or(0) as usize,
                        ui_monitoring: r["ui_monitoring"].as_u64().unwrap_or(0) as usize,
                        accessibility: r["accessibility"].as_u64().unwrap_or(0) as usize,
                    }
                }
                _ => DbRowCounts::default(),
            };

            if args.json {
                let out = StatusOutput {
                    connected: true,
                    url: url.clone(),
                    status: health.status.clone(),
                    frame_status: health.frame_status.clone(),
                    audio_status: health.audio_status.clone(),
                    monitors,
                    audio_devices,
                    db_rows,
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                println!("\nScreenpipe Status");
                println!("{:=<60}", "");
                println!("  URL:            {}", url);
                println!("  Status:         {}", health.status);
                if let Some(ref fs) = health.frame_status {
                    println!("  Frames:         {}", fs);
                }
                if let Some(ref a) = health.audio_status {
                    println!("  Audio:          {}", a);
                }
                if let Some(ref ts) = health.last_frame_timestamp {
                    println!("  Last frame:     {}", ts);
                }
                if let Some(ref ts) = health.last_audio_timestamp {
                    println!("  Last audio:     {}", ts);
                }
                if let Some(ref msg) = health.message {
                    println!("  Message:        {}", msg);
                }

                if !monitors.is_empty() {
                    println!("\n  Monitors:");
                    for m in &monitors {
                        println!(
                            "    {} — {}x{}{}",
                            m.name,
                            m.width,
                            m.height,
                            if m.is_default { " (default)" } else { "" }
                        );
                    }
                }

                if !audio_devices.is_empty() {
                    println!("\n  Audio Devices:");
                    for d in &audio_devices {
                        println!(
                            "    {}{}",
                            d.name,
                            if d.is_default { " (default)" } else { "" }
                        );
                    }
                }

                let total = db_rows.frames + db_rows.ocr + db_rows.audio
                    + db_rows.ui_monitoring + db_rows.accessibility;
                println!("\n  Database:");
                println!("    Frames:         {}", db_rows.frames);
                println!("    OCR texts:      {}", db_rows.ocr);
                println!("    Audio:          {}", db_rows.audio);
                println!("    UI monitoring:  {}", db_rows.ui_monitoring);
                println!("    Accessibility:  {}", db_rows.accessibility);
                println!("    Total rows:     {}", total);

                println!("{:=<60}\n", "");
            }
        }
        None => {
            if args.json {
                let out = StatusOutput {
                    connected: false,
                    url: url.clone(),
                    status: "unreachable".into(),
                    frame_status: None,
                    audio_status: None,
                    monitors: vec![],
                    audio_devices: vec![],
                    db_rows: DbRowCounts::default(),
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
            } else {
                eprintln!("Screenpipe is not reachable at {}", url);
                eprintln!("Install: curl -fsSL get.screenpi.pe/cli | sh");
                eprintln!("Start:   screenpipe");
            }
        }
    }
    Ok(())
}
