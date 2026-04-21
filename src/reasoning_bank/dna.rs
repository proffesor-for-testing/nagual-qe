//! Pattern DNA visualization barcode.
//!
//! Encodes pattern attributes into a compact visual "DNA barcode" using Unicode
//! characters. Each segment encodes one attribute (domain, reward, age, reuse count,
//! surprise score, tier) producing a human-readable fingerprint at a glance.
//!
//! # Example
//!
//! ```ignore
//! use nagual::reasoning_bank::dna::PatternDNA;
//! use nagual::reasoning_bank::PatternTier;
//!
//! let barcode = PatternDNA::encode("rust", 0.85, 3, 12, 0.6, PatternTier::Crystal);
//! // => "💎 ▐ ●●●●○ 🟢 ×10"
//!
//! let compact = PatternDNA::to_compact("rust", 0.85, 3, 12, 0.6, PatternTier::Crystal);
//! // => "💎●●●●○🟢"
//!
//! let html = PatternDNA::to_html("rust", 0.85, 3, 12, 0.6, PatternTier::Crystal);
//! // => "<span ...>...</span>"
//! ```

use super::PatternTier;

/// Pattern DNA visualization barcode generator.
///
/// Encodes pattern attributes into Unicode visual representations for
/// quick human recognition of pattern characteristics.
pub struct PatternDNA;

impl PatternDNA {
    /// Encode pattern attributes into a full DNA barcode string.
    ///
    /// The barcode consists of segments separated by spaces:
    /// - Tier icon (prefix)
    /// - Domain glyph
    /// - Reward blocks (5 positions, filled/empty)
    /// - Age indicator (freshness emoji)
    /// - Reuse multiplier
    ///
    /// # Arguments
    ///
    /// * `domain` - The pattern domain (e.g., "rust", "python", "security")
    /// * `reward` - Reward score from 0.0 to 1.0
    /// * `age_days` - Age of the pattern in days
    /// * `reuse_count` - Number of times the pattern has been reused
    /// * `surprise_score` - Novelty score from 0.0 to 1.0
    /// * `tier` - Pattern confidence tier
    ///
    /// # Returns
    ///
    /// A String containing the visual DNA barcode.
    pub fn encode(
        domain: &str,
        reward: f32,
        age_days: i64,
        reuse_count: u32,
        surprise_score: f32,
        tier: PatternTier,
    ) -> String {
        let tier_icon = Self::tier_icon(tier);
        let domain_glyph = Self::domain_glyph(domain);
        let reward_blocks = Self::reward_blocks(reward);
        let age_indicator = Self::age_indicator(age_days);
        let reuse_label = Self::reuse_label(reuse_count);
        let surprise_bar = Self::surprise_bar(surprise_score);

        format!(
            "{} {} {} {} {} {}",
            tier_icon, domain_glyph, reward_blocks, age_indicator, reuse_label, surprise_bar
        )
    }

    /// Return a compact version of the DNA barcode (tier icon + reward blocks + age indicator).
    ///
    /// Shorter than the full `encode` output, suitable for inline display in tables
    /// or constrained terminal widths.
    pub fn to_compact(
        _domain: &str,
        reward: f32,
        age_days: i64,
        _reuse_count: u32,
        _surprise_score: f32,
        tier: PatternTier,
    ) -> String {
        let tier_icon = Self::tier_icon(tier);
        let reward_blocks = Self::reward_blocks(reward);
        let age_indicator = Self::age_indicator(age_days);

        format!("{}{}{}", tier_icon, reward_blocks, age_indicator)
    }

    /// Return an HTML representation of the DNA barcode with inline CSS styling.
    ///
    /// Each segment is wrapped in a `<span>` with appropriate color styling.
    pub fn to_html(
        domain: &str,
        reward: f32,
        age_days: i64,
        reuse_count: u32,
        surprise_score: f32,
        tier: PatternTier,
    ) -> String {
        let tier_icon = Self::tier_icon(tier);
        let tier_color = match tier {
            PatternTier::Reflex => "#ffd700",
            PatternTier::Crystal => "#00bfff",
            PatternTier::Booster => "#90ee90",
        };

        let domain_glyph = Self::domain_glyph(domain);

        let reward_blocks = Self::reward_blocks(reward);
        let reward_color = if reward >= 0.8 {
            "#00cc00"
        } else if reward >= 0.5 {
            "#cccc00"
        } else {
            "#cc0000"
        };

        let age_indicator = Self::age_indicator(age_days);
        let reuse_label = Self::reuse_label(reuse_count);
        let surprise_bar = Self::surprise_bar(surprise_score);

        format!(
            "<span style=\"font-family:monospace\">\
             <span style=\"color:{}\">{}</span> \
             <span>{}</span> \
             <span style=\"color:{}\">{}</span> \
             <span>{}</span> \
             <span>{}</span> \
             <span>{}</span>\
             </span>",
            tier_color, tier_icon, domain_glyph, reward_color, reward_blocks,
            age_indicator, reuse_label, surprise_bar
        )
    }

    /// Get the tier icon prefix.
    fn tier_icon(tier: PatternTier) -> &'static str {
        match tier {
            PatternTier::Booster => "\u{1F680}",  // rocket
            PatternTier::Crystal => "\u{1F48E}",  // gem stone
            PatternTier::Reflex => "\u{26A1}",    // high voltage / lightning
        }
    }

    /// Map a domain string to a Unicode block character.
    fn domain_glyph(domain: &str) -> &'static str {
        // Use the root domain (before the first dot) for mapping.
        let root = domain.split('.').next().unwrap_or(domain);
        match root.to_lowercase().as_str() {
            "rust" => "\u{2590}",         // RIGHT HALF BLOCK ▐
            "python" => "\u{258C}",       // LEFT HALF BLOCK ▌
            "javascript" | "js" => "\u{2588}", // FULL BLOCK █
            "typescript" | "ts" => "\u{2593}", // DARK SHADE ▓
            "go" | "golang" => "\u{2592}",     // MEDIUM SHADE ▒
            "java" => "\u{2591}",         // LIGHT SHADE ░
            "security" => "\u{2584}",     // LOWER HALF BLOCK ▄
            "performance" => "\u{2580}",  // UPPER HALF BLOCK ▀
            "testing" => "\u{259A}",      // QUADRANT UPPER LEFT AND LOWER RIGHT ▚
            "resilience" => "\u{259E}",   // QUADRANT UPPER RIGHT AND LOWER LEFT ▞
            _ => "\u{2596}",              // QUADRANT LOWER LEFT ▖
        }
    }

    /// Encode a reward value (0.0-1.0) as 5 filled/empty circles.
    fn reward_blocks(reward: f32) -> String {
        let clamped = reward.clamp(0.0, 1.0);
        let filled = (clamped * 5.0).round() as usize;
        let empty = 5 - filled;
        let mut s = String::with_capacity(20);
        for _ in 0..filled {
            s.push('\u{25CF}'); // BLACK CIRCLE ●
        }
        for _ in 0..empty {
            s.push('\u{25CB}'); // WHITE CIRCLE ○
        }
        s
    }

    /// Map age in days to a freshness indicator.
    fn age_indicator(age_days: i64) -> &'static str {
        if age_days < 7 {
            "\u{1F7E2}" // GREEN CIRCLE 🟢 fresh
        } else if age_days < 30 {
            "\u{1F7E1}" // YELLOW CIRCLE 🟡 recent
        } else if age_days <= 90 {
            "\u{26AA}"  // WHITE CIRCLE ⚪ middle
        } else {
            "\u{1F534}" // RED CIRCLE 🔴 stale
        }
    }

    /// Encode reuse count into a multiplier label.
    fn reuse_label(reuse_count: u32) -> String {
        if reuse_count >= 20 {
            "\u{00D7}20+".to_string()
        } else if reuse_count >= 10 {
            "\u{00D7}10".to_string()
        } else if reuse_count >= 5 {
            "\u{00D7}5".to_string()
        } else {
            format!("\u{00D7}{}", reuse_count)
        }
    }

    /// Encode surprise score (0.0-1.0) as a small bar.
    fn surprise_bar(surprise_score: f32) -> String {
        let clamped = surprise_score.clamp(0.0, 1.0);
        let filled = (clamped * 3.0).round() as usize;
        let empty = 3 - filled;
        let mut s = String::from("S:");
        for _ in 0..filled {
            s.push('\u{2588}'); // FULL BLOCK █
        }
        for _ in 0..empty {
            s.push('\u{2591}'); // LIGHT SHADE ░
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_booster_tier_returns_rocket_icon() {
        let barcode = PatternDNA::encode("rust", 0.5, 10, 3, 0.3, PatternTier::Booster);
        assert!(
            barcode.contains('\u{1F680}'),
            "Booster tier should contain rocket icon, got: {}",
            barcode
        );
    }

    #[test]
    fn test_encode_crystal_tier_returns_diamond_icon() {
        let barcode = PatternDNA::encode("python", 0.7, 5, 8, 0.5, PatternTier::Crystal);
        assert!(
            barcode.contains('\u{1F48E}'),
            "Crystal tier should contain diamond icon, got: {}",
            barcode
        );
    }

    #[test]
    fn test_encode_reflex_tier_returns_lightning_icon() {
        let barcode = PatternDNA::encode("go", 0.9, 2, 25, 0.8, PatternTier::Reflex);
        assert!(
            barcode.contains('\u{26A1}'),
            "Reflex tier should contain lightning icon, got: {}",
            barcode
        );
    }

    #[test]
    fn test_encode_high_reward_returns_five_filled_blocks() {
        let barcode = PatternDNA::encode("rust", 0.95, 1, 5, 0.5, PatternTier::Booster);
        // 0.95 * 5 = 4.75 rounds to 5 filled
        let filled_count = barcode.matches('\u{25CF}').count();
        assert_eq!(
            filled_count, 5,
            "Reward 0.95 should produce 5 filled circles, got {} in: {}",
            filled_count, barcode
        );
    }

    #[test]
    fn test_encode_zero_reward_returns_empty_blocks() {
        let barcode = PatternDNA::encode("rust", 0.0, 1, 0, 0.0, PatternTier::Booster);
        let empty_count = barcode.matches('\u{25CB}').count();
        assert_eq!(
            empty_count, 5,
            "Reward 0.0 should produce 5 empty circles, got {} in: {}",
            empty_count, barcode
        );
    }

    #[test]
    fn test_to_compact_shorter_than_full_encode() {
        let full = PatternDNA::encode("rust", 0.7, 15, 8, 0.5, PatternTier::Crystal);
        let compact = PatternDNA::to_compact("rust", 0.7, 15, 8, 0.5, PatternTier::Crystal);
        assert!(
            compact.len() < full.len(),
            "Compact ({} bytes) should be shorter than full ({} bytes)",
            compact.len(),
            full.len()
        );
    }

    #[test]
    fn test_to_html_contains_span_tags() {
        let html = PatternDNA::to_html("python", 0.6, 20, 3, 0.4, PatternTier::Booster);
        assert!(
            html.contains("<span"),
            "HTML output should contain <span tags, got: {}",
            html
        );
        assert!(
            html.contains("</span>"),
            "HTML output should contain closing </span> tags, got: {}",
            html
        );
    }

    #[test]
    fn test_age_fresh_less_than_7_days() {
        let barcode = PatternDNA::encode("rust", 0.5, 3, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{1F7E2}'),
            "Age <7d should show green circle, got: {}",
            barcode
        );
    }

    #[test]
    fn test_age_recent_less_than_30_days() {
        let barcode = PatternDNA::encode("rust", 0.5, 15, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{1F7E1}'),
            "Age 7-29d should show yellow circle, got: {}",
            barcode
        );
    }

    #[test]
    fn test_age_middle_less_than_90_days() {
        let barcode = PatternDNA::encode("rust", 0.5, 60, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{26AA}'),
            "Age 30-90d should show white circle, got: {}",
            barcode
        );
    }

    #[test]
    fn test_age_stale_over_90_days() {
        let barcode = PatternDNA::encode("rust", 0.5, 120, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{1F534}'),
            "Age >90d should show red circle, got: {}",
            barcode
        );
    }

    #[test]
    fn test_reuse_label_high() {
        let barcode = PatternDNA::encode("rust", 0.5, 10, 25, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains("\u{00D7}20+"),
            "Reuse >=20 should show x20+, got: {}",
            barcode
        );
    }

    #[test]
    fn test_reuse_label_medium() {
        let barcode = PatternDNA::encode("rust", 0.5, 10, 10, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains("\u{00D7}10"),
            "Reuse >=10 should show x10, got: {}",
            barcode
        );
    }

    #[test]
    fn test_reuse_label_low() {
        let barcode = PatternDNA::encode("rust", 0.5, 10, 2, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains("\u{00D7}2"),
            "Reuse count 2 should show x2, got: {}",
            barcode
        );
    }

    #[test]
    fn test_domain_glyph_rust() {
        let barcode = PatternDNA::encode("rust", 0.5, 1, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{2590}'),
            "Rust domain should use RIGHT HALF BLOCK, got: {}",
            barcode
        );
    }

    #[test]
    fn test_domain_glyph_python() {
        let barcode = PatternDNA::encode("python", 0.5, 1, 1, 0.1, PatternTier::Booster);
        assert!(
            barcode.contains('\u{258C}'),
            "Python domain should use LEFT HALF BLOCK, got: {}",
            barcode
        );
    }

    #[test]
    fn test_to_html_contains_style_attribute() {
        let html = PatternDNA::to_html("rust", 0.9, 5, 10, 0.7, PatternTier::Reflex);
        assert!(
            html.contains("style="),
            "HTML output should contain style attributes, got: {}",
            html
        );
    }

    #[test]
    fn test_to_html_contains_monospace_font() {
        let html = PatternDNA::to_html("rust", 0.5, 10, 3, 0.3, PatternTier::Booster);
        assert!(
            html.contains("monospace"),
            "HTML output should specify monospace font, got: {}",
            html
        );
    }

    #[test]
    fn test_encode_mid_reward_returns_three_filled() {
        let barcode = PatternDNA::encode("rust", 0.5, 1, 1, 0.0, PatternTier::Booster);
        // 0.5 * 5 = 2.5, rounds to 3
        let filled = barcode.matches('\u{25CF}').count();
        assert_eq!(
            filled, 3,
            "Reward 0.5 should produce 3 filled circles (round 2.5), got {}",
            filled
        );
    }

    #[test]
    fn test_surprise_bar_encoding() {
        let barcode = PatternDNA::encode("rust", 0.5, 1, 1, 1.0, PatternTier::Booster);
        assert!(
            barcode.contains("S:"),
            "Barcode should contain surprise bar prefix, got: {}",
            barcode
        );
    }
}
