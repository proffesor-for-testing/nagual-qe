-- PostgreSQL graduation triggers for automatic tier promotion
-- Depends on: 015_pattern_tiers (tier column must exist)

-- Function to check and apply tier promotion on pattern update
CREATE OR REPLACE FUNCTION check_pattern_graduation()
RETURNS TRIGGER AS $$
BEGIN
    -- Check for Reflex promotion: reward >= 0.9 AND reuse_count >= 20
    IF NEW.reward >= 0.9 AND NEW.reuse_count >= 20 AND
       (OLD.tier IS NULL OR OLD.tier != 'reflex') THEN
        NEW.tier = 'reflex';
        PERFORM pg_notify('nagual_pattern_promoted',
            json_build_object(
                'id', NEW.id,
                'old_tier', COALESCE(OLD.tier, 'booster'),
                'new_tier', 'reflex',
                'reward', NEW.reward,
                'reuse_count', NEW.reuse_count
            )::text);
    -- Check for Crystal promotion: reward >= 0.7 AND reuse_count >= 5
    ELSIF NEW.reward >= 0.7 AND NEW.reuse_count >= 5 AND
          (OLD.tier IS NULL OR OLD.tier = 'booster') THEN
        NEW.tier = 'crystal';
        PERFORM pg_notify('nagual_pattern_promoted',
            json_build_object(
                'id', NEW.id,
                'old_tier', COALESCE(OLD.tier, 'booster'),
                'new_tier', 'crystal',
                'reward', NEW.reward,
                'reuse_count', NEW.reuse_count
            )::text);
    -- Check for demotion from Reflex
    ELSIF OLD.tier = 'reflex' AND NEW.reward < 0.8 THEN
        NEW.tier = 'crystal';
        PERFORM pg_notify('nagual_pattern_promoted',
            json_build_object(
                'id', NEW.id,
                'old_tier', 'reflex',
                'new_tier', 'crystal',
                'reward', NEW.reward,
                'reuse_count', NEW.reuse_count
            )::text);
    -- Check for demotion from Crystal
    ELSIF OLD.tier = 'crystal' AND NEW.reward < 0.6 THEN
        NEW.tier = 'booster';
        PERFORM pg_notify('nagual_pattern_promoted',
            json_build_object(
                'id', NEW.id,
                'old_tier', 'crystal',
                'new_tier', 'booster',
                'reward', NEW.reward,
                'reuse_count', NEW.reuse_count
            )::text);
    END IF;

    -- Notify on any pattern insert for real-time dashboards
    IF TG_OP = 'INSERT' THEN
        PERFORM pg_notify('nagual_pattern_stored',
            json_build_object(
                'id', NEW.id,
                'category', NEW.category,
                'tier', COALESCE(NEW.tier, 'booster')
            )::text);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger fires on INSERT and UPDATE of reward or reuse_count columns
CREATE TRIGGER trg_pattern_graduation
    BEFORE INSERT OR UPDATE OF reward, reuse_count ON reasoning_patterns
    FOR EACH ROW EXECUTE FUNCTION check_pattern_graduation();

-- Notification for consolidation events
CREATE OR REPLACE FUNCTION notify_consolidation()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        PERFORM pg_notify('nagual_consolidation_complete',
            json_build_object(
                'deleted_id', OLD.id,
                'category', OLD.category
            )::text);
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_pattern_consolidation_notify
    AFTER DELETE ON reasoning_patterns
    FOR EACH ROW EXECUTE FUNCTION notify_consolidation();
