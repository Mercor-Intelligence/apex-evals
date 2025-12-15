-- ============================================================
-- DROP AND CREATE ALL TABLES FOR ACE EVALUATION BENCHMARK
-- ============================================================
-- This script drops existing tables and recreates them with fresh schema.
-- WARNING: This will DELETE ALL EXISTING DATA in these tables!
-- Total: 10 models × 4 domains × 2 tables = 80 tables
-- ============================================================

DO $$
DECLARE
    model_name TEXT;
    domain_name TEXT;
    domain_display TEXT;
    table_name TEXT;
    sql_statement TEXT;
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
            -- Set display name for Criterion Type column
            domain_display := CASE domain_name
                WHEN 'shopping' THEN 'Shopping'
                WHEN 'food' THEN 'Food'
                WHEN 'gaming' THEN 'Gaming'
                WHEN 'diy' THEN 'DIY'
            END;
            
            -- ============================================================
            -- DROP AND CREATE TASKS TABLE
            -- ============================================================
            table_name := 'tasks_' || domain_name || '_' || model_name;
            
            -- Drop existing table (CASCADE removes dependent objects like RLS policies)
            EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', table_name);
            RAISE NOTICE 'Dropped table: %', table_name;
            
            -- Create fresh table
            sql_statement := 'CREATE TABLE "' || table_name || '" ('
                || '"Task ID" INTEGER PRIMARY KEY, '
                || '"Prompt" TEXT, '
                || '"Specified Prompt" TEXT, '
                || '"Workflow" TEXT, '
                || '"Criteria List" JSONB';
            
            -- Add Shop vs. Product column for Shopping domain only
            IF domain_name = 'shopping' THEN
                sql_statement := sql_statement || ', "Shop vs. Product" TEXT';
            END IF;
            
            -- Add columns for runs 1-8
            FOR i IN 1..8 LOOP
                sql_statement := sql_statement 
                    || ', "Response Text - ' || i || '" TEXT'
                    || ', "Product Source Map - ' || i || '" JSONB'
                    || ', "Grounding Source Meta Data - ' || i || '" JSONB'
                    || ', "Direct Grounding - ' || i || '" JSONB'
                    || ', "Scores - ' || i || '" JSONB'
                    || ', "Total Score - ' || i || '" NUMERIC'
                    || ', "Total Hurdle Score - ' || i || '" NUMERIC'
                    || ', "Score Overview - ' || i || '" JSONB'
                    || ', "Failed Grounded Sites - ' || i || '" JSONB';
            END LOOP;
            
            sql_statement := sql_statement || ')';
            
            -- Execute the CREATE TABLE statement
            EXECUTE sql_statement;
            RAISE NOTICE 'Created table: %', table_name;
            
            -- ============================================================
            -- DROP AND CREATE CRITERIA TABLE
            -- ============================================================
            table_name := 'criteria_' || domain_name || '_' || model_name;
            
            -- Drop existing table (CASCADE removes dependent objects like RLS policies)
            EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', table_name);
            RAISE NOTICE 'Dropped table: %', table_name;
            
            -- Create fresh table
            sql_statement := 'CREATE TABLE "' || table_name || '" ('
                || '"Criterion ID" INTEGER PRIMARY KEY, '
                || '"Task ID" INTEGER, '
                || '"Prompt" TEXT, '
                || '"Description" TEXT, '
                || '"Criterion Grounding Check" TEXT, '
                || '"Hurdle Tag" TEXT, '
                || '"Criterion Type (' || domain_display || ')" TEXT, '
                || '"Specified Prompt" TEXT, '
                || '"Workflow" TEXT';
            
            -- Add Shop vs. Product column for Shopping domain only
            IF domain_name = 'shopping' THEN
                sql_statement := sql_statement || ', "Shop vs. Product" TEXT';
            END IF;
            
            -- Add columns for runs 1-8
            FOR i IN 1..8 LOOP
                sql_statement := sql_statement 
                    || ', "Score - ' || i || '" INTEGER'
                    || ', "Reasoning - ' || i || '" JSONB'
                    || ', "Failure Step - ' || i || '" TEXT';
            END LOOP;
            
            sql_statement := sql_statement || ')';
            
            -- Execute the CREATE TABLE statement
            EXECUTE sql_statement;
            RAISE NOTICE 'Created table: %', table_name;
            
        END LOOP;
    END LOOP;
    
    RAISE NOTICE 'Successfully dropped and recreated all 80 tables!';
    RAISE NOTICE 'WARNING: All existing data was deleted.';
    RAISE NOTICE 'Next step: Run create_rls_policies.sql to enable RLS and create policies.';
END $$;
