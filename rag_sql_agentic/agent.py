from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from .llm import LLMModel
import re
import sqlite3



class SQLAgent:
    """
    Agent that queries a SQL database to answer questions.
    """
    def __init__(self, db_path: str, llm_model: LLMModel):
        self.db_path = db_path
        self.llm_model = llm_model
        
        # Connect to SQLite DB
        # include_tables can be used if we only want specific tables
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
        
        # Create Agent
        # Uses the exposed LangChain LLM from LLMModel
        

        self.prompt = """You are an SQL expert.

        You have direct access to ONE SQLite database.

        You MUST answer the user's question by querying the database.

        RULES:

        - Always use the database to answer.

        - Create valid SQLite statements.

        - ONLY use tables and columns that exist in the database.

        - NEVER guess the schema.

        - NEVER ask the user for clarification.

        - NEVER explain SQL statements unless explicitly asked.

        - If the question cannot be answered from the database, say:

        "I cannot answer this question using the current database."

        \n

        PROCESS:

        1. Check the database schema if necessary.

        2. Write an exact SQL query.

        3. Execute the query.

        4. Return the final answer in natural language.

        \n

        You are not allowed to modify the database.

        Read queries only.
        """

        self.agent_executor = create_sql_agent(
            llm=self.llm_model.llm,
            db=self.db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            prefix = self.prompt
        )

    def is_ready(self) -> bool:
        """Return True if the sqlite DB file contains at least one user table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            rows = cur.fetchall()
            conn.close()
            return len(rows) > 0
        except Exception:
            return False

    def _extract_final_answer(self, text: str) -> str:
        """Extract the user-facing final answer from agent output.

        Looks for phrases like 'Final Answer:' or 'Answer:' and returns everything
        after that. Strips code fences (```...```) and a leading 'sql' tag if present.
        If no marker is found, returns the original text trimmed.
        """
        if not text:
            return ""
        # Find 'Final Answer:' or 'Answer:' (case-insensitive)
        m = re.search(r"(?:Final Answer:|Answer:)\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            # Remove triple-backtick code fences and optional 'sql' marker
            ans = re.sub(r"```\s*(?:sql)?\s*", "", ans, flags=re.IGNORECASE)
            ans = re.sub(r"```", "", ans)
            ans = re.sub(r"^\s*sql\s+", "", ans, flags=re.IGNORECASE)
            return ans.strip()
        # fallback: return original text trimmed
        return text.strip()

    def answer(self, query: str):
        # Execute the agent and return only the final natural-language answer
        try:
            response = self.agent_executor.invoke(
                {"input": query},
                {"callbacks": None} # ensure no extra logs
            )
            raw = response.get("output", "") if isinstance(response, dict) else str(response)
            return self._extract_final_answer(raw)
        except Exception as e:
            # If it's an output parsing error, try to extract from the exception message as a last resort
            msg = str(e)
            extracted = self._extract_final_answer(msg)
            if extracted:
                return extracted
            return f"Error: {msg}"
