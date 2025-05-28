#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from typing import Dict, List, Any, Tuple
import openai
import backoff
import sqlparse
from tqdm import tqdm
import spacy
import re
openai.debug = True

# Initialize SpaCy NLP model
nlp = spacy.load("en_core_web_sm")

import logging
from contextlib import contextmanager
from threading import Timer

logging.basicConfig(level=logging.INFO)

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@contextmanager
def time_limit(seconds):
    """
    Context manager to limit the execution time of a block of code.
    It raises a TimeoutException if the block takes longer than 'seconds'.
    """
    # Register the signal function handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Schedule the signal after `seconds` seconds
    signal.alarm(seconds)
    
    try:
        yield
    except TimeoutException:
        logging.error("Operation timed out")
        raise
    finally:
        # Cancel the alarm
        signal.alarm(0)



def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

def knowledge_package(data_json, knowledge=False):
    knowledge_list = []
    for data in data_json:
        knowledge_list.append(data['evidence'])

    return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = f"{db_root_path}/{data['db_id']}/{data['db_id']}.sqlite"  # Adjusted here
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['evidence'])
    
    return question_list, db_path_list, knowledge_list

class SchemaCache:
    def __init__(self, cache_dir: str = "./schema_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def get_cache_file_path(self, db_path: str) -> str:
        """Generate a cache file path based on the database file path."""
        cache_file_name = os.path.basename(db_path).replace('.sqlite', '.json')
        return os.path.join(self.cache_dir, cache_file_name)

    def load_schema_from_cache(self, db_path: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Load schema from cache if it exists."""
        cache_file = self.get_cache_file_path(db_path)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data['schemas'], data['all_columns']
        return None, None

    def save_schema_to_cache(self, db_path: str, schemas: Dict[str, Any], all_columns: Dict[str, List[str]]):
        """Save the schema to the cache."""
        cache_file = self.get_cache_file_path(db_path)
        data = {
            'schemas': schemas,
            'all_columns': all_columns
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def get_or_fetch_schema(self, db_path: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Return the schema from cache or fetch from database if not cached."""
        # Try loading from cache first
        schemas, all_columns = self.load_schema_from_cache(db_path)
        if schemas and all_columns:
            print(f"Loaded schema from cache for {db_path}")
            return schemas, all_columns

        # If not cached, fetch from database and save to cache
        schemas, all_columns = get_db_schemas(db_path)
        self.save_schema_to_cache(db_path, schemas, all_columns)
        return schemas, all_columns


schema_cache = SchemaCache()

def clean_sql_query(sql: str) -> str:
    sql = re.sub(r'^```sql|```$', '', sql).strip()
    if not re.match(r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\b', sql, re.IGNORECASE):
        sql = 'SELECT ' + sql
    return sql

def get_db_schemas(db_path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
        cursor = conn.cursor()
        
        # Get all the table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schemas = {}
        all_columns = {}
        
        for table in tables:
            table_name = table[0]
            
            # Get the CREATE statement for each table
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            schema_result = cursor.fetchone()
            if schema_result:
                schemas[table_name] = schema_result[0]
                
                # Get the columns for the table
                cursor.execute(f'PRAGMA table_info("[{table_name}]")')
                columns = cursor.fetchall()
                all_columns[table_name] = [col[1] for col in columns]  # Extract column names

        return schemas, all_columns

def generate_schema_prompt(db_path, relevant_tables=None, relevant_columns=None, attention_weights=None, num_rows=None):
    """
    Generates the schema prompt with CREATE statements, example rows, and attention weights
    for each table and column in the database.
    
    :param db_path: Path to the database file
    :param relevant_tables: List of relevant tables based on question entities
    :param relevant_columns: List of relevant columns based on question entities
    :param attention_weights: Dictionary of attention weights for tables and columns
    :param num_rows: Number of example rows to include in the prompt
    :return: A string representing the schema prompt with attention weights
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}

    # Loop through tables in the schema
    for table in tables:
        if table[0] == 'sqlite_sequence':
            continue
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table[0]}'")
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt

        # Add attention weight to the table (if applicable)
        if relevant_tables and attention_weights:
            table_weight = attention_weights.get(table[0], 0.5)  # Default weight of 0.5 for non-relevant tables
            create_prompt += f" -- Attention Weight: {table_weight}"
        else:
            table_weight = 0.5  # Default attention weight for non-relevant tables

        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = f"`{cur_table}`"
            cursor.execute(f"SELECT * FROM {cur_table} LIMIT {num_rows}")
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {cur_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
            schemas[table[0]] = f"{create_prompt} \n {verbose_prompt}"

        # Add attention weight to columns (if applicable)
        if relevant_columns and attention_weights:
            for column in relevant_columns:
                if column[0] == table[0]:  # Check if the column belongs to the current table
                    column_weight = attention_weights.get(column[1], 0.5)  # Default weight of 0.5 for non-relevant columns
                    create_prompt += f"\n    {column[1]} -- Attention Weight: {column_weight}"

    # Format and combine all table schemas into a single prompt
    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)
    return schema_prompt

def detailed_self_discovery_guide():
    """
    Provides detailed step-by-step guidance for constructing SQL queries.
    """
    steps = [
        "-- Step 1: Analyze the schema and identify all tables and columns relevant to the question.",
        "-- Step 2: Determine how these tables are related. Identify the type of joins needed to connect these tables.",
        "-- Step 3: Outline what information you need to extract from each table. Consider what columns need to be selected or calculated.",
        "-- Step 4: Identify any filters or conditions that need to be applied to the data. This includes WHERE clauses and any necessary data transformations.",
        "-- Step 5: Consider if any data needs to be aggregated. Determine what GROUP BY or HAVING clauses might be necessary.",
        "-- Step 6: Plan how the results should be ordered. Decide on the ORDER BY clauses.",
        "-- Step 7: Construct the SQL query step by step, incorporating the elements identified in the previous steps."
    ]
    return '\n'.join(steps)

def generate_sql_file(sql_lst, output_path=None):
    result = {}
    for i, sql in enumerate(sql_lst):
        result[i] = sql
    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    return result

def generate_comment_prompt(question, knowledge=None):
    """
    Generates the comment prompt for SQL query generation based on the question and optional external knowledge.
    """
    pattern_prompt_no_kg = "-- Using valid SQLite, answer the following questions for the tables provided above."
    pattern_prompt_kg = "-- Using valid SQLite and understanding External Knowledge, answer the following questions for the tables provided above."
    question_prompt = f"-- {question}"
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    return pattern_prompt_kg + '\n' + question_prompt if knowledge else pattern_prompt_no_kg + '\n' + question_prompt


def cot_wizard():
    """
    Generates the Chain-of-Thought prompting.
    """
    cot = "\nGenerate the SQL after thinking step by step: "
    return cot

def nice_look_table(column_names: list, values: list):
    """
    Create a nicely formatted table for example rows in the schema.
    """
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output


def escape_column_name(column_name: str) -> str:
    """
    Escapes column names with double quotes for SQLite.
    Double quotes are used as they are the SQL standard for identifier quoting.
    """
    # Check if the column is already escaped (either with double quotes or backticks)
    if not (column_name.startswith('"') and column_name.endswith('"')) and not (column_name.startswith('`') and column_name.endswith('`')):
        return f'"{column_name}"'
    return column_name

def escape_all_column_names(sql_query: str, all_columns: Dict[str, List[str]]) -> str:
    """
    Escapes all column names in the given SQL query using exact matching.
    
    :param sql_query: The SQL query string
    :param all_columns: A dictionary where keys are table names and values are lists of column names
    :return: The SQL query with all column names properly escaped
    """
    # Tokenize the query by splitting on spaces and punctuation to avoid partial matches
    words = re.split(r'(\W+)', sql_query)  # Split by non-word characters while keeping them in the result

    # Flatten the list of all column names from all tables
    column_names = {col for columns in all_columns.values() for col in columns}

    # Escape the column names that are in the list of columns
    escaped_words = [escape_column_name(word) if word in column_names else word for word in words]

    # Rejoin the tokenized query back into a single string
    return ''.join(escaped_words)


def is_valid_sql(sql_query):
    try:
        parsed = sqlparse.parse(sql_query)
        return len(parsed) > 0 and parsed[0].get_type() == 'SELECT'
    except:
        return False

def execute_and_validate_query(db_path: str, sql_query: str, question: str, all_columns: Dict[str, List[str]]) -> Dict[str, Any]:
    validation_result = {"execution_success": False, "row_count": 0}  # Initialize with row_count
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch columns from cache
        _, all_columns = schema_cache.get_or_fetch_schema(db_path)

        # Escape all column names in the SQL query
        escaped_sql_query = escape_all_column_names(sql_query, all_columns)

        cursor.execute(escaped_sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

        validation_result.update({
            "execution_success": True,
            "row_count": len(results),
            "column_count": len(column_names),
            "sample_data": results[:5] if results else [],
            "column_names": column_names,
            "potential_issues": []
        })

        # Check for empty results
        if len(results) == 0:
            validation_result["potential_issues"].append("Query returned no results")

        # Check for excessive results
        if len(results) > 1000:
            validation_result["potential_issues"].append("Query returned an unusually large number of rows")

        # Check if all columns are the same value (might indicate a mistake in JOIN conditions)
        if results and all(len(set(row)) == 1 for row in results):
            validation_result["potential_issues"].append("All columns have the same value, possible JOIN issue")

        return validation_result

    except sqlite3.Error as e:
        validation_result["error_message"] = str(e)
        return validation_result

    finally:
        if conn:
            conn.close()


def generate_feedback_from_validation(validation_result: Dict[str, Any]) -> str:
    """
    Generates feedback based on query execution validation.
    """
    feedback = []

    # Handle execution errors
    if not validation_result.get("execution_success", False):
        feedback.append(f"The query failed to execute. Error: {validation_result.get('error_message', 'Unknown error')}")

    # Safely access row_count and add feedback if necessary
    if validation_result.get("row_count", 0) == 0:
        feedback.append("The query returned no results. Possible issues with WHERE conditions or table/column names.")
    elif validation_result.get("row_count", 0) > 1000:
        feedback.append("The query returned a large number of rows. Consider adding more specific conditions.")

    # Include potential issues from the validation_result
    if "potential_issues" in validation_result and validation_result["potential_issues"]:
        for issue in validation_result["potential_issues"]:
            feedback.append(f"- {issue}")

    return "\n".join(feedback)


def regenerate_sql_with_feedback(question: str, db_path: str, feedback: str, attempts_history: List[Dict], all_columns: Dict[str, List[str]]) -> str:
    # Use schema cache
    schema, _ = schema_cache.get_or_fetch_schema(db_path)

    prompt_content = f"""Given the following question, database schema, and feedback, generate an improved SQL query:

Question: {question}

Schema:
{schema}

Previous attempt feedback:
{feedback}

Attempts history:
{attempts_history}

Improved SQL query (no explanations or comments, just SQL):
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an SQL expert. Respond only with valid SQL queries, no explanations or comments."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=200,
            temperature=0.3
        )
        raw_sql = response['choices'][0]['message']['content'].strip()
        escaped_sql = escape_all_column_names(raw_sql, all_columns)
        return escaped_sql
    except Exception as e:
        return f"error: {e}"


def extract_entities_and_relationships(question: str):
    """
    Extracts key entities and their relationships from the question using SpaCy.
    """
    doc = nlp(question)
    entities = [ent.text for ent in doc.ents]  # Extract named entities
    relationships = []
    for token in doc:
        if token.dep_ in ('nsubj', 'dobj'):  # Subject-object dependencies
            relationships.append((token.text, token.head.text))
    return entities, relationships


def map_entities_to_schema(entities: List[str], relationships: List[tuple], schema: Dict[str, str]):
    """
    Maps extracted entities and their relationships to schema tables and columns.
    Returns a list of relevant tables, columns, and attention weights.
    
    :param entities: List of entities extracted from the question.
    :param relationships: List of tuples representing relationships between entities.
    :param schema: Dictionary representing the schema (tables and columns).
    :return: Tuple of relevant tables, relevant columns, and attention weights.
    """
    relevant_tables = []
    relevant_columns = []
    attention_weights = {}

    # Map entities to tables and columns
    for entity in entities:
        entity_lower = entity.lower()
        for table, create_stmt in schema.items():
            if entity_lower in table.lower():
                relevant_tables.append(table)
                attention_weights[table] = attention_weights.get(table, 1.0) + 1.0
            else:
                # Look for columns containing the entity
                for column in create_stmt.split('(')[1].split(')')[0].split(','):
                    column_name = column.strip().split()[0]
                    if entity_lower in column_name.lower():
                        relevant_columns.append((table, column_name))
                        attention_weights[column_name] = attention_weights.get(column_name, 1.0) + 1.0

    # Handle relationships (if necessary)
    for rel in relationships:
        subject, obj = rel  # Extract subject and object of the relationship
        subject_lower, obj_lower = subject.lower(), obj.lower()

        # If both subject and object are found in schema, increase attention weight for related tables/columns
        for table, create_stmt in schema.items():
            if subject_lower in table.lower() or obj_lower in table.lower():
                attention_weights[table] = attention_weights.get(table, 1.0) + 0.5  # Increase weight for table

            for column in create_stmt.split('(')[1].split(')')[0].split(','):
                column_name = column.strip().split()[0]
                if subject_lower in column_name.lower() or obj_lower in column_name.lower():
                    attention_weights[column_name] = attention_weights.get(column_name, 1.0) + 0.5  # Increase weight for columns

    return relevant_tables, relevant_columns, attention_weights



def create_attention_mask(schema: Dict[str, str], relevant_tables: List[str], relevant_columns: List[tuple], attention_weights: Dict[str, float]):
    """
    Creates an attention mask that assigns higher weights to relevant schema elements.
    Returns a formatted string for inclusion in the prompt.
    """
    attention_info = {}

    for table, create_stmt in schema.items():
        table_weight = attention_weights.get(table, 0.5)
        attention_info[table] = table_weight

        columns = create_stmt.split('(')[1].split(')')[0].split(',')
        for column in columns:
            column_name = column.strip().split()[0]
            full_column = f"{table}.{column_name}"
            if (table, column_name) in relevant_columns:
                attention_info[full_column] = attention_weights.get(column_name, 1.0)
            else:
                attention_info[full_column] = 0.3  # Lower weight for non-relevant columns

    attention_mask_str = "### Schema Attention Weights ###\n"
    for element, weight in attention_info.items():
        attention_mask_str += f"{element}: {weight}\n"

    return attention_mask_str


def generate_difficulty_based_prompt(db_path, question, knowledge=None, difficulty='simple', attempts_history=[], schema=None, chain_of_thought='False'):
    """
    Generates a prompt based on the difficulty level of the question, incorporating schema details,
    external knowledge, and specific guidance for handling the question based on its difficulty.
    """
    # Use the cache to fetch the schema
    schema_dict, all_columns = schema_cache.get_or_fetch_schema(db_path)

    # Pre-process question to extract entities and relationships
    entities, relationships = extract_entities_and_relationships(question)

    # Map entities to schema elements
    relevant_tables, relevant_columns, attention_weights = map_entities_to_schema(entities, relationships, schema_dict)

    # Generate schema prompt
    schema = generate_schema_prompt(db_path, relevant_tables, relevant_columns, attention_weights)

    # Generate attention mask
    attention_mask = create_attention_mask(schema_dict, relevant_tables, relevant_columns, attention_weights)

    # Generate few-shot examples based on whether external knowledge is used
    if knowledge:
        few_shot_example = few_shot()  # With external knowledge
    else:
        few_shot_example = few_shot_no_kg()  # Without external knowledge

    # Generate the main question prompt
    comment_prompt = generate_comment_prompt(question, knowledge)
    history_summary = "\n\nPrevious Attempts Summary:\n" if attempts_history else ""

    for idx, attempt in enumerate(attempts_history, start=1):
        history_summary += f"Attempt {idx}: Generated SQL: {attempt['generated_sql']}, Success: {attempt['is_successful']}\n"

    cot_prompt = cot_wizard() if chain_of_thought == 'True' else ""  # Include chain-of-thought prompting only if enabled

    if difficulty == 'simple':
        detailed_steps = detailed_self_discovery_guide()
    else:
        detailed_steps = detailed_self_discovery_guide() if difficulty in ['moderate', 'challenging'] else ""

    final_instruction = "Write the SQL query to answer the question, prioritizing schema elements based on the attention weights provided."

    # Combine all elements into the final prompt
    prompt = (
        f"{few_shot_example}\n\n"  # Add the few-shot example
        f"{schema}\n\n"
        f"{comment_prompt}\n\n"
        f"{cot_prompt}\n\n"  # Chain-of-thought section (included if enabled)
        f"{detailed_steps}\n"
        f"{history_summary}\n"
        f"{attention_mask}\n\n"
        f"{final_instruction}\n"
    )

    return prompt


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)

def connect_gpt(engine, prompt, max_tokens, temperature, stop):
    try:
        # Call OpenAI's ChatCompletion.create() directly
        result = openai.ChatCompletion.create(
            model=engine,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are an SQL expert. Respond only with valid SQL queries, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stop=stop  # Ensure stop is a list of strings
        )

        # Extract and return the raw SQL from the response
        raw_sql = result['choices'][0]['message']['content'].strip()
        return raw_sql

    except Exception as e:
        # Return the error message for debugging
        return f'error: {str(e)}'


def calculate_confidence(validation_result: Dict[str, Any]) -> float:
    """
    Calculate the confidence score of a generated SQL query based on validation feedback.
    """
    confidence = 0.0

    # Execution success increases confidence
    if validation_result.get("execution_success"):
        confidence += 0.5

        # Row count feedback
        row_count = validation_result.get("row_count", 0)
        if 1 <= row_count <= 100:  # Ideal range of rows returned
            confidence += 0.3
        elif row_count > 1000:  # Too many rows
            confidence -= 0.2

        # Check for potential issues
        potential_issues = validation_result.get("potential_issues", [])
        if not potential_issues:  # No issues increases confidence
            confidence += 0.2
        else:
            confidence -= 0.1 * len(potential_issues)

    return max(0.0, min(confidence, 1.0))

import time
@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=5)
def retry_connect_gpt(*args, **kwargs):
    return connect_gpt(*args, **kwargs)

def collect_response_from_gpt_with_retry(db_path_list, question_list, api_key, engine, knowledge_list=None):
    openai.api_key = api_key
    response_list = []
    feedback_results = {}

    schema_cache = SchemaCache()

    for i, question in enumerate(question_list):
        logging.info(f"Processing {i + 1}/{len(question_list)}: {question}")
        start_time = time.time()

        try:
            with time_limit(120):  # 10 minutes timeout for the entire question processing
                schema_dict, all_columns = schema_cache.get_or_fetch_schema(db_path_list[i])
                logging.info(f"Schema loaded for question {i + 1}")

                attempts_history = []
                attempt = 0
                is_successful = False
                best_confidence_score = 0.0
                best_sql_query = None

                while attempt < 3 and not is_successful:
                    logging.info(f"Attempt {attempt + 1} for question: {question}")
                    
                    # Generate the prompt based on the current attempt and knowledge (if available)
                    cur_prompt = generate_difficulty_based_prompt(
                        db_path=db_path_list[i],
                        question=question,
                        knowledge=knowledge_list[i] if knowledge_list else None,
                        schema=schema_dict
                    )

                    # Call GPT with a timeout
                    try:
                        with time_limit(90):  # 2 minutes timeout for API call
                            result = retry_connect_gpt(engine=engine, prompt=cur_prompt, max_tokens=256, temperature=0.5, stop=[';'])
                    except TimeoutException:
                        logging.error(f"API call timed out for attempt {attempt + 1}")
                        attempt += 1
                        continue  # Move to the next attempt if there's a timeout

                    sql = clean_sql_query(result.strip())
                    if not sql.upper().startswith('SELECT'):
                        sql = f'SELECT {sql}'

                    # Validate the generated SQL query
                    try:
                        with time_limit(60):  # 1 minute timeout for query execution
                            validation_result = execute_and_validate_query(db_path_list[i], sql, question, all_columns)
                    except TimeoutException:
                        logging.error(f"Query execution timed out for attempt {attempt + 1}")
                        attempt += 1
                        continue  # Move to the next attempt if there's a timeout

                    confidence_score = calculate_confidence(validation_result)

                    if not validation_result["execution_success"]:
                        feedback = generate_feedback_from_validation(validation_result)
                        sql = regenerate_sql_with_feedback(question, db_path_list[i], feedback, attempts_history, all_columns)

                        try:
                            with time_limit(60):  # 1 minute timeout for regenerated query execution
                                validation_result = execute_and_validate_query(db_path_list[i], sql, question, all_columns)
                        except TimeoutException:
                            logging.error(f"Regenerated query execution timed out for attempt {attempt + 1}")
                            attempt += 1
                            continue  # Move to the next attempt if there's a timeout

                        confidence_score = calculate_confidence(validation_result)

                    attempts_history.append({
                        "generated_sql": sql,
                        "is_successful": validation_result["execution_success"],
                        "feedback": validation_result.get("potential_issues", []),
                        "confidence_score": confidence_score
                    })

                    # Keep track of the best query
                    if confidence_score > best_confidence_score:
                        best_confidence_score = confidence_score
                        best_sql_query = sql

                    if validation_result["execution_success"] and confidence_score >= 0.7:
                        is_successful = True

                    attempt += 1

                response_list.append(best_sql_query if best_sql_query else sql)

                feedback_results[i] = {
                    "question": question,
                    "best_sql_query": best_sql_query,
                    "best_confidence_score": best_confidence_score,
                    "attempts": attempt,
                    "feedback_history": attempts_history
                }

        except TimeoutException:
            logging.error(f"Processing timed out for question: {question}")
            response_list.append(None)
            feedback_results[i] = {
                "question": question,
                "error": "Processing timed out"
            }
            continue  # Skip to the next question if there's a timeout

        except Exception as e:
            logging.error(f"Error processing question: {question}")
            logging.error(f"Error details: {str(e)}")
            response_list.append(None)
            feedback_results[i] = {
                "question": question,
                "error": str(e)
            }
            continue  # Skip to the next question if there's an exception

        end_time = time.time()
        logging.info(f"Question {i + 1} processed in {end_time - start_time:.2f} seconds")

    return response_list, feedback_results


def save_feedback(feedback_results, feedback_output_path):
    """
    Saves the feedback results to a specified file.
    """
    if feedback_output_path:
        directory_path = os.path.dirname(feedback_output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(feedback_output_path, 'w') as json_file:
            json.dump(feedback_results, json_file, indent=4)


if __name__ == '__main__':
    # Argument parsing
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--eval_path', type=str, default='')
    args_parser.add_argument('--mode', type=str, default='dev')
    args_parser.add_argument('--use_knowledge', type=str, default='False')
    args_parser.add_argument('--db_root_path', type=str, default='')
    args_parser.add_argument('--api_key', type=str, required=True)
    args_parser.add_argument('--engine', type=str, required=True, default='code-davinci-002')
    args_parser.add_argument('--data_output_path', type=str)
    args_parser.add_argument('--feedback_output_path', type=str, help="Path to store feedback output", default=None)
    args_parser.add_argument('--chain_of_thought', type=str, default='False', help="Enable chain-of-thought prompting (True/False)")  # New argument

    args = args_parser.parse_args()

    # Load evaluation data
    eval_data = json.load(open(args.eval_path, 'r'))

    # Decouple question and schema
    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data, db_root_path=args.db_root_path)
    if args.use_knowledge == 'True':
        responses, feedback_results = collect_response_from_gpt_with_retry(
            db_path_list=db_path_list,
            question_list=question_list,
            api_key=args.api_key,
            engine=args.engine,
            #force_reload=True  # Enable force reload
    )
    
    else:
        responses, feedback_results = collect_response_from_gpt_with_retry(
        db_path_list=db_path_list,
        question_list=question_list,
        api_key=args.api_key,
        engine=args.engine,
        #force_reload=True 
    )
    response_list = [clean_sql_query(sql) for sql in responses]  # Clean all SQL queries here
    output_name = args.data_output_path + 'predict_' + args.mode + '.json'
    generate_sql_file(sql_lst=response_list, output_path=output_name)
    save_feedback(feedback_results, args.feedback_output_path)
    print(f'Successfully collected results from {args.engine} for {args.mode} evaluation.')

