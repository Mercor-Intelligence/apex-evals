-- ============================================================
-- ENABLE RLS AND CREATE POLICIES FOR ALL ACE TABLES
-- ============================================================
-- This script:
-- 1. Enables Row Level Security on all tables
-- 2. Creates policies allowing full access for all users
-- Total: 10 models × 4 domains × 2 tables = 80 tables
-- ============================================================

DO $$
DECLARE
    model_name TEXT;
    domain_name TEXT;
    table_name TEXT;
    policy_name TEXT;
BEGIN
    -- List of all models
    FOREACH model_name IN ARRAY ARRAY[
        'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-3-pro',
        'gpt-5', 'gpt-5.1', 'o3', 'o3-pro',
        'sonnet-4.5', 'opus-4.1', 'opus-4.5'
    ]
    LOOP
        -- List of all domains
        FOREACH domain_name IN ARRAY ARRAY['shopping', 'food', 'gaming', 'diy']
        LOOP
            -- ============================================================
            -- TASKS TABLE
            -- ============================================================
            table_name := 'tasks_' || domain_name || '_' || model_name;
            policy_name := 'Allow all operations on ' || table_name;
            
            -- 1. Drop existing policy if it exists
            BEGIN
                EXECUTE format('DROP POLICY IF EXISTS %I ON %I', policy_name, table_name);
            EXCEPTION WHEN OTHERS THEN
                -- Ignore errors if policy doesn't exist
                NULL;
            END;
            
            -- 2. Enable RLS
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
            RAISE NOTICE 'Enabled RLS on: %', table_name;
            
            -- 3. Create policy
            EXECUTE format(
                'CREATE POLICY %I ON %I FOR ALL USING (true) WITH CHECK (true)',
                policy_name,
                table_name
            );
            RAISE NOTICE 'Created policy: %', policy_name;
            
            -- ============================================================
            -- CRITERIA TABLE
            -- ============================================================
            table_name := 'criteria_' || domain_name || '_' || model_name;
            policy_name := 'Allow all operations on ' || table_name;
            
            -- 1. Drop existing policy if it exists
            BEGIN
                EXECUTE format('DROP POLICY IF EXISTS %I ON %I', policy_name, table_name);
            EXCEPTION WHEN OTHERS THEN
                -- Ignore errors if policy doesn't exist
                NULL;
            END;
            
            -- 2. Enable RLS
            EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', table_name);
            RAISE NOTICE 'Enabled RLS on: %', table_name;
            
            -- 3. Create policy
            EXECUTE format(
                'CREATE POLICY %I ON %I FOR ALL USING (true) WITH CHECK (true)',
                policy_name,
                table_name
            );
            RAISE NOTICE 'Created policy: %', policy_name;
            
        END LOOP;
    END LOOP;
    
    RAISE NOTICE 'Successfully enabled RLS and created policies for all 80 tables!';
END $$;
