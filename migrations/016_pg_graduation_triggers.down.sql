-- Remove graduation triggers and functions
DROP TRIGGER IF EXISTS trg_pattern_consolidation_notify ON reasoning_patterns;
DROP FUNCTION IF EXISTS notify_consolidation();
DROP TRIGGER IF EXISTS trg_pattern_graduation ON reasoning_patterns;
DROP FUNCTION IF EXISTS check_pattern_graduation();
