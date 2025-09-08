import asyncio
import time
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule
from rich.tree import Tree
from rich.status import Status
from rich.markdown import Markdown
import google.generativeai as genai
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from threading import Lock
import threading
import hashlib
import mimetypes
import base64
from rich.filesize import decimal
# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('multi_agent_assistant.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
class AgentRole(Enum):
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"

@dataclass
class AgentResponse:
    agent_role: AgentRole
    content: str
    confidence: float
    suggestions: List[str]
    errors_found: List[str]
    improvements: List[str]
    metadata: Dict[str, Any]

    def to_dict(self):
        return {
            'agent_role': self.agent_role.value,
            'content': self.content,
            'confidence': self.confidence,
            'suggestions': self.suggestions,
            'errors_found': self.errors_found,
            'improvements': self.improvements,
            'metadata': self.metadata
        }

@dataclass
class MultiAgentSession:
    session_id: str
    user_query: str
    iterations: List[Dict[str, Any]]
    final_response: str
    quality_score: float
    total_iterations: int
    timestamp: datetime
    context_used: str

    def to_dict(self):
        serializable_iterations = []
        for iteration in self.iterations:
            serializable_iteration = iteration.copy()
            for key in ['generator_response', 'analyzer_response', 'optimizer_response']:
                if key in serializable_iteration and hasattr(serializable_iteration[key], 'to_dict'):
                    serializable_iteration[key] = serializable_iteration[key].to_dict()
                elif key in serializable_iteration and isinstance(serializable_iteration[key], dict):
                    if 'agent_role' in serializable_iteration[key] and hasattr(serializable_iteration[key]['agent_role'], 'value'):
                        serializable_iteration[key]['agent_role'] = serializable_iteration[key]['agent_role'].value
            serializable_iterations.append(serializable_iteration)
        return {
            'session_id': self.session_id,
            'user_query': self.user_query,
            'iterations': serializable_iterations,
            'final_response': self.final_response,
            'quality_score': self.quality_score,
            'total_iterations': self.total_iterations,
            'timestamp': self.timestamp.isoformat(),
            'context_used': self.context_used
        }

@dataclass
class Conversation:
    id: str
    user_id: str
    timestamp: datetime
    user_message: str
    ai_response: str
    multi_agent_session: Optional['MultiAgentSession']
    context: str
    sentiment: str
    topics: List[str]
    potential_followups: List[str]

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'user_message': self.user_message,
            'ai_response': self.ai_response,
            'multi_agent_session': self.multi_agent_session.to_dict() if self.multi_agent_session else None,
            'context': self.context,
            'sentiment': self.sentiment,
            'topics': self.topics,
            'potential_followups': self.potential_followups
        }

    @classmethod
    def from_dict(cls, data: Dict):
        multi_agent_data = data.get('multi_agent_session')
        multi_agent_session = None
        if multi_agent_data:
            multi_agent_session = MultiAgentSession(
                session_id=multi_agent_data['session_id'],
                user_query=multi_agent_data['user_query'],
                iterations=multi_agent_data['iterations'],
                final_response=multi_agent_data['final_response'],
                quality_score=multi_agent_data['quality_score'],
                total_iterations=multi_agent_data['total_iterations'],
                timestamp=datetime.fromisoformat(multi_agent_data['timestamp']),
                context_used=multi_agent_data.get('context_used', '')
            )
        return cls(
            id=data['id'],
            user_id=data['user_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_message=data['user_message'],
            ai_response=data['ai_response'],
            multi_agent_session=multi_agent_session,
            context=data['context'],
            sentiment=data['sentiment'],
            topics=data['topics'],
            potential_followups=data['potential_followups']
        )

@dataclass
class Task:
    id: str
    conversation_id: str
    task_type: str
    description: str
    priority: int
    status: str
    created_at: datetime
    scheduled_for: datetime
    result: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'task_type': self.task_type,
            'description': self.description,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'scheduled_for': self.scheduled_for.isoformat(),
            'result': self.result,
            'metadata': self.metadata
        }

class Document:
        id: str
        filename: str
        content: str
        file_type: str
        file_size: int
        upload_time: datetime
        user_id: str
        hash_value: str
        metadata: Dict[str, Any]

        def to_dict(self):
            return {
                'id': self.id,
                'filename': self.filename,
                'content': self.content,
                'file_type': self.file_type,
                'file_size': self.file_size,
                'upload_time': self.upload_time.isoformat(),
                'user_id': self.user_id,
                'hash_value': self.hash_value,
                'metadata': self.metadata
            }

        @classmethod
        def from_dict(cls, data: Dict):
            return cls(
                id=data['id'],
                filename=data['filename'],
                content=data['content'],
                file_type=data['file_type'],
                file_size=data['file_size'],
                upload_time=datetime.fromisoformat(data['upload_time']),
                user_id=data['user_id'],
                hash_value=data['hash_value'],
                metadata=data.get('metadata', {})
            )

# --- File Manager ---
class TextFileManager:
    def __init__(self, data_dir: str = "multi_agent_data"):
        self.data_dir = Path(data_dir)
        self.file_lock = Lock()
        self.data_dir.mkdir(exist_ok=True)
        self.conversations_file = self.data_dir / "conversations.json"
        self.tasks_file = self.data_dir / "tasks.json"
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        for file_path in [self.conversations_file, self.tasks_file]:
            if not file_path.exists():
                file_path.write_text("[]")

    def _read_json_file(self, file_path: Path) -> List[Dict]:
        try:
            with self.file_lock:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    return json.loads(content) if content else []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            backup_path = file_path.with_suffix('.backup')
            file_path.rename(backup_path)
            return []
        except FileNotFoundError:
            return []

    def _write_json_file(self, file_path: Path, data: List[Dict]):
        with self.file_lock:
            try:
                json_content = json.dumps(data, indent=2, ensure_ascii=False)
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(json_content)
                temp_path.replace(file_path)
            except Exception as e:
                logger.error(f"Failed to write {file_path}: {e}")
                raise

    def save_conversation(self, conversation: Conversation):
        try:
            conversations = self._read_json_file(self.conversations_file)
            conversations = [c for c in conversations if c.get('id') != conversation.id]
            conversations.append(conversation.to_dict())
            conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            self._write_json_file(self.conversations_file, conversations)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    def save_task(self, task: Task):
        try:
            tasks = self._read_json_file(self.tasks_file)
            tasks = [t for t in tasks if t.get('id') != task.id]
            tasks.append(task.to_dict())
            tasks.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            self._write_json_file(self.tasks_file, tasks)
        except Exception as e:
            logger.error(f"Failed to save task: {e}")

    def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Conversation]:
        conversations_data = self._read_json_file(self.conversations_file)
        user_conversations = [c for c in conversations_data if c.get('user_id') == user_id]
        user_conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return [Conversation.from_dict(conv_data) for conv_data in user_conversations[:limit]]

    def get_conversation_context(self, user_id: str, limit: int = 15) -> str:
        conversations = self.get_recent_conversations(user_id, limit)
        if not conversations:
            return "No previous conversation history available."
        context_parts = ["=== PREVIOUS CONVERSATION HISTORY ==="]
        for i, conv in enumerate(conversations):
            context_parts.append(f"\n--- Conversation {i + 1} ({conv.timestamp.strftime('%Y-%m-%d %H:%M')}) ---")
            context_parts.append(f"USER: {conv.user_message}")
            context_parts.append(f"ASSISTANT: {conv.ai_response}")
            if conv.multi_agent_session:
                context_parts.append(f"[Quality: {conv.multi_agent_session.quality_score:.2f}, Iterations: {conv.multi_agent_session.total_iterations}]")
            if conv.topics:
                context_parts.append(f"[Topics: {', '.join(conv.topics)}]")
        context_parts.append("\n=== END OF CONVERSATION HISTORY ===")
        return "\n".join(context_parts)

    def get_pending_tasks(self) -> List[Task]:
        tasks_data = self._read_json_file(self.tasks_file)
        now = datetime.now()
        pending_tasks = []
        for task_data in tasks_data:
            try:
                if task_data.get('status') == 'pending' and datetime.fromisoformat(task_data.get('scheduled_for', '')) <= now:
                    pending_tasks.append(Task.from_dict(task_data))
            except Exception as e:
                logger.warning(f"Skipping malformed task data: {e}")
        pending_tasks.sort(key=lambda x: (-x.priority, x.scheduled_for))
        return pending_tasks

    def get_storage_stats(self) -> Dict[str, Any]:
        conversations_data = self._read_json_file(self.conversations_file)
        tasks_data = self._read_json_file(self.tasks_file)
        documents_data = self._read_json_file(self.documents_file)  # ADD THIS LINE

        user_stats = {}
        for conv in conversations_data:
            user_id = conv.get('user_id', 'unknown')
            user_stats[user_id] = user_stats.get(user_id, 0) + 1

        return {
            'total_conversations': len(conversations_data),
            'total_tasks': len(tasks_data),
            'total_documents': len(documents_data),  # ADD THIS LINE
            'users': user_stats,
            'oldest_conversation': min([c.get('timestamp', '') for c in conversations_data], default='none'),
            'newest_conversation': max([c.get('timestamp', '') for c in conversations_data], default='none'),
            'files_exist': {
                'conversations': self.conversations_file.exists(),
                'tasks': self.tasks_file.exists(),
                'documents': self.documents_file.exists()  # ADD THIS LINE
            }
        }

    def __init__(self, data_dir: str = "multi_agent_data"):
        self.data_dir = Path(data_dir)
        self.file_lock = Lock()
        self.data_dir.mkdir(exist_ok=True)
        self.conversations_file = self.data_dir / "conversations.json"
        self.tasks_file = self.data_dir / "tasks.json"
        self.documents_file = self.data_dir / "documents.json"  # ADD THIS LINE
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        for file_path in [self.conversations_file, self.tasks_file, self.documents_file]:  # ADD documents_file
            if not file_path.exists():
                file_path.write_text("[]")

    # ADD THESE NEW METHODS TO TextFileManager:

    def save_document(self, document: Document):
        try:
            documents = self._read_json_file(self.documents_file)
            documents = [d for d in documents if d.get('id') != document.id]
            documents.append(document.to_dict())
            documents.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
            self._write_json_file(self.documents_file, documents)
            logger.info(f"Document saved: {document.filename}")
        except Exception as e:
            logger.error(f"Failed to save document: {e}")

    def get_user_documents(self, user_id: str, limit: int = 20) -> List[Document]:
        documents_data = self._read_json_file(self.documents_file)
        user_docs = [d for d in documents_data if d.get('user_id') == user_id]
        user_docs.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        return [Document.from_dict(doc_data) for doc_data in user_docs[:limit]]

    def delete_document(self, document_id: str, user_id: str) -> bool:
        try:
            documents = self._read_json_file(self.documents_file)
            original_count = len(documents)
            documents = [d for d in documents if not (d.get('id') == document_id and d.get('user_id') == user_id)]
            if len(documents) < original_count:
                self._write_json_file(self.documents_file, documents)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_documents_context(self, user_id: str, limit: int = 5) -> str:
        documents = self.get_user_documents(user_id, limit)
        if not documents:
            return ""

        context_parts = ["=== UPLOADED DOCUMENTS CONTEXT ==="]
        for i, doc in enumerate(documents):
            context_parts.append(f"\n--- Document {i + 1}: {doc.filename} ---")
            # Truncate content to reasonable size for context
            content_preview = doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content
            context_parts.append(content_preview)
        context_parts.append("\n=== END OF DOCUMENTS CONTEXT ===")
        return "\n".join(context_parts)

    def process_uploaded_file(self, file_path: str, user_id: str) -> Optional[Document]:
        """Process an uploaded file and return a Document object."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None

            # Read file content
            content = ""
            file_type = mimetypes.guess_type(file_path)[0] or 'text/plain'

            if file_type.startswith('text/'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                # For non-text files, store basic info
                content = f"[Binary file: {file_path.name}]"

            # Calculate hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # Get file size
            file_size = file_path.stat().st_size

            document = Document(
                id=f"{user_id}_{int(time.time())}_{file_hash[:8]}",
                filename=file_path.name,
                content=content,
                file_type=file_type,
                file_size=file_size,
                upload_time=datetime.now(),
                user_id=user_id,
                hash_value=file_hash,
                metadata={
                    'original_path': str(file_path),
                    'processed_at': datetime.now().isoformat()
                }
            )

            self.save_document(document)
            return document

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

# --- AI Agent ---
class AIAgent:
    def __init__(self, api_key: str, role: AgentRole, temperature: float = 0.7):
        genai.configure(api_key=api_key)
        self.role = role
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

    async def process(self, user_query: str, context: str = "", previous_iteration: Dict = None) -> AgentResponse:
        if self.role == AgentRole.GENERATOR:
            return await self._generate_solution(user_query, context)
        elif self.role == AgentRole.ANALYZER:
            return await self._analyze_solution(user_query, context, previous_iteration)
        elif self.role == AgentRole.OPTIMIZER:
            return await self._optimize_solution(user_query, context, previous_iteration)

    async def _generate_solution(self, user_query: str, context: str) -> AgentResponse:
        prompt = f"""
        Role: Solution Generator Agent
        Task: Provide a comprehensive initial solution to the user's query using conversation history.
        {context}
        Current User Query: {user_query}
        Instructions:
        Instructions:
        1. Use the conversation history to inform your response but do not mention it explicitly
        2. Provide a detailed, well-structured response that naturally incorporates relevant context
        3. Do not reference previous conversations or mention using conversation history
        4. Make the response feel complete and self-contained
        5. Include practical examples where relevant
        6. Consider multiple approaches if applicable
        7. Be thorough but clear
        8. Rate your confidence in this solution (0-1)
        """
        try:
            response = await self.model.generate_content_async(prompt)
            return AgentResponse(
                agent_role=self.role,
                content=response.text,
                confidence=0.8,
                suggestions=[],
                errors_found=[],
                improvements=[],
                metadata={}
            )
        except Exception as e:
            print(f"Error in _generate_solution: {e}")
            return AgentResponse(
                agent_role=self.role,
                content=f"Error generating solution: {e}",
                confidence=0.5,
                suggestions=[],
                errors_found=[str(e)],
                improvements=[],
                metadata={"error": str(e)}
            )
    async def _analyze_solution(self, user_query: str, context: str, previous_iteration: Dict) -> AgentResponse:
        """Agent 2: Analyze solution considering conversation history."""
        generator_response = previous_iteration.get("generator_response", {})
        solution = generator_response.get("content", "")

        prompt = f"""
        Role: Critical Analyzer Agent
        Task: Analyze the provided solution for errors, gaps, and improvements, considering conversation history.

        {context}

        Original User Query: {user_query}

        Solution to Analyze:
        {solution}

        Instructions:
        1. Review the conversation history to understand the full context
        2. Check if the solution properly addresses the user's query in context of previous conversations
        3. Identify factual errors or inaccuracies
        4. Find logical inconsistencies
        5. Spot missing information or gaps
        6. Check if the solution maintains conversational continuity
        7. Verify if previous relevant information was properly considered
        8. Suggest areas for improvement
        9. Rate the overall quality (0-1)
        10. Be thorough but constructive

        Format your response as JSON:
        {{
            "analysis": "Your detailed analysis",
            "errors_found": ["error1", "error2"],
            "gaps_identified": ["gap1", "gap2"],
            "improvements_needed": ["improvement1", "improvement2"],
            "quality_score": 0.75,
            "strengths": ["strength1", "strength2"],
            "recommendations": ["rec1", "rec2"],
            "context_adherence": 0.8,
            "continuity_score": 0.7
        }}
        """

        try:
            response = await self.model.generate_content_async(prompt)
            response_text = response.text.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]

            data = json.loads(response_text)

            return AgentResponse(
                agent_role=self.role,
                content=data.get("analysis", response_text),
                confidence=data.get("quality_score", 0.7),
                suggestions=data.get("recommendations", []),
                errors_found=data.get("errors_found", []),
                improvements=data.get("improvements_needed", []),
                metadata=data
            )
        except Exception as e:
            logger.error(f"Analyzer agent error: {e}")
            return AgentResponse(
                agent_role=self.role,
                content="Analysis completed with some limitations.",
                confidence=0.6,
                suggestions=[],
                errors_found=[],
                improvements=[],
                metadata={"error": str(e)}
            )

    async def _optimize_solution(self, user_query: str, context: str, previous_iteration: Dict) -> AgentResponse:
        generator_response = previous_iteration.get("generator_response", {})
        analyzer_response = previous_iteration.get("analyzer_response", {})
        original_solution = generator_response.get("content", "")
        analysis = analyzer_response.get("content", "")
        errors = analyzer_response.get("errors_found", [])
        improvements = analyzer_response.get("improvements", [])
        prompt = f"""
        Role: Solution Optimizer Agent
        Task: Create an improved solution based on analysis feedback and conversation history.
        {context}
        Original User Query: {user_query}
        Original Solution: {original_solution}
        Analysis Feedback: {analysis}
        Identified Errors: {errors}
        Suggested Improvements: {improvements}
        Instructions:
        1. Review the conversation history to maintain context and continuity
        2. Fix all identified errors
        3. Address the gaps and improvements
        4. Enhance clarity and completeness
        5. Provide a significantly improved solution
        Format your response as JSON:
        {{
            "optimized_solution": "Your improved solution here",
            "changes_made": ["change1", "change2"],
            "errors_fixed": ["fix1", "fix2"],
            "enhancements": ["enhancement1", "enhancement2"],
            "confidence": 0.95,
            "context_integration": "How previous conversations were integrated"
        }}
        """
        try:
            response = await self.model.generate_content_async(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            data = json.loads(response_text)
            return AgentResponse(
                agent_role=self.role,
                content=data.get("optimized_solution", response_text),
                confidence=data.get("confidence", 0.9),
                suggestions=data.get("enhancements", []),
                errors_found=[],
                improvements=data.get("changes_made", []),
                metadata=data
            )
        except Exception as e:
            logger.error(f"Optimizer agent error: {e}")
            return AgentResponse(
                agent_role=self.role,
                content=original_solution,
                confidence=0.7,
                suggestions=[],
                errors_found=[],
                improvements=[],
                metadata={"error": str(e)}
            )

# --- Multi-Agent System ---
class MultiAgentSystem:
    def __init__(self, api_key: str):
        self.generator = AIAgent(api_key, AgentRole.GENERATOR, temperature=0.8)
        self.analyzer = AIAgent(api_key, AgentRole.ANALYZER, temperature=0.3)
        self.optimizer = AIAgent(api_key, AgentRole.OPTIMIZER, temperature=0.5)
        self.max_iterations = 3
        self.quality_threshold = 0.85

    async def process_query(self, user_query: str, context: str = "") -> MultiAgentSession:
        session_id = f"session_{int(time.time())}"
        iterations = []
        current_iteration = 0
        final_response = ""
        quality_score = 0.0
        while current_iteration < self.max_iterations:
            iteration_data = {"iteration": current_iteration + 1, "timestamp": datetime.now().isoformat()}
            generator_response = await self.generator.process(user_query, context, iterations[-1] if iterations else None)
            iteration_data["generator_response"] = asdict(generator_response)
            analyzer_response = await self.analyzer.process(user_query, context, iteration_data)
            iteration_data["analyzer_response"] = asdict(analyzer_response)
            optimizer_response = await self.optimizer.process(user_query, context, iteration_data)
            iteration_data["optimizer_response"] = asdict(optimizer_response)
            iterations.append(iteration_data)
            final_response = optimizer_response.content
            quality_score = optimizer_response.confidence
            if quality_score >= self.quality_threshold or not analyzer_response.errors_found:
                break
            current_iteration += 1
        return MultiAgentSession(
            session_id=session_id,
            user_query=user_query,
            iterations=iterations,
            final_response=final_response,
            quality_score=quality_score,
            total_iterations=len(iterations),
            timestamp=datetime.now(),
            context_used=context[:500] + "..." if len(context) > 500 else context
        )

# --- Autonomous Assistant ---
class AutonomousMultiAgentAssistant:
    def __init__(self, gemini_api_key: str):
        self.file_manager = TextFileManager()
        self.multi_agent_system = MultiAgentSystem(gemini_api_key)
        self.running = False
        self.task_executor_thread = None

    def start(self):
        self.running = True
        self.task_executor_thread = threading.Thread(target=self._task_executor_loop, daemon=True)
        self.task_executor_thread.start()

    def stop(self):
        self.running = False
        if self.task_executor_thread:
            self.task_executor_thread.join(timeout=5)

    async def process_user_message(self, user_id: str, message: str) -> str:
        context = self.file_manager.get_conversation_context(user_id, 15)
        multi_agent_session = await self.multi_agent_system.process_query(message, context)
        conversation_id = f"{user_id}_{int(time.time())}"
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            timestamp=datetime.now(),
            user_message=message,
            ai_response=multi_agent_session.final_response,
            multi_agent_session=multi_agent_session,
            context=context[:2000] + "..." if len(context) > 2000 else context,
            sentiment="neutral",
            topics=[],
            potential_followups=[]
        )
        self.file_manager.save_conversation(conversation)
        await self._analyze_and_create_tasks(conversation)
        return multi_agent_session.final_response

    async def _analyze_and_create_tasks(self, conversation: Conversation):
        suggested_tasks = []
        if "smart home" in conversation.user_message.lower():
            suggested_tasks = [
                "Research latest smart thermostat models and energy savings",
                "Compare smart lighting systems within  dollar 500-800 budget range"
            ]
        elif "sensors" in conversation.user_message.lower():
            suggested_tasks = [
                "Research motion sensors for smart lighting automation",
                "Compare ambient light sensors for energy efficiency"
            ]
        for i, task_desc in enumerate(suggested_tasks):
            task_id = f"{conversation.id}_task_{i}"
            schedule_delay = timedelta(minutes=2 + (i * 3))
            task = Task(
                id=task_id,
                conversation_id=conversation.id,
                task_type="research",
                description=task_desc,
                priority=2,
                status="pending",
                created_at=datetime.now(),
                scheduled_for=datetime.now() + schedule_delay,
                metadata={"user_id": conversation.user_id}
            )
            self.file_manager.save_task(task)

    def _task_executor_loop(self):
        while self.running:
            try:
                pending_tasks = self.file_manager.get_pending_tasks()
                for task in pending_tasks[:2]:
                    asyncio.run(self._execute_task_async(task))
                time.sleep(45)
            except Exception as e:
                logger.error(f"Task executor error: {e}")
                time.sleep(60)

    async def _execute_task_async(self, task: Task):
        try:
            task.status = "in_progress"
            self.file_manager.save_task(task)
            context = ""
            if task.metadata and 'user_id' in task.metadata:
                context = self.file_manager.get_conversation_context(task.metadata['user_id'], 3)
            session = await self.multi_agent_system.process_query(f"Research and provide information about: {task.description}", context)
            task.result = session.final_response
            task.status = "completed"
            task.metadata["quality_score"] = session.quality_score
            task.metadata["iterations"] = session.total_iterations
            self.file_manager.save_task(task)
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            self.file_manager.save_task(task)

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        conversations = self.file_manager.get_recent_conversations(user_id, 20)
        if not conversations:
            return {"message": "No conversation history found"}
        quality_scores = []
        total_iterations = []
        context_usage = []
        for conv in conversations:
            if conv.multi_agent_session:
                quality_scores.append(conv.multi_agent_session.quality_score)
                total_iterations.append(conv.multi_agent_session.total_iterations)
                context_usage.append(len(conv.context) > 100)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_iterations = sum(total_iterations) / len(total_iterations) if total_iterations else 0
        context_usage_rate = sum(context_usage) / len(context_usage) if context_usage else 0
        return {
            "total_conversations": len(conversations),
            "multi_agent_sessions": len([c for c in conversations if c.multi_agent_session]),
            "average_quality_score": round(avg_quality, 2),
            "average_iterations": round(avg_iterations, 1),
            "highest_quality": max(quality_scores) if quality_scores else 0,
            "context_usage_rate": round(context_usage_rate * 100, 1),
            "topics": self._extract_topics(conversations),
            "last_interaction": conversations[0].timestamp.isoformat() if conversations else None,
            "conversation_file_status": "Active" if self.file_manager.conversations_file.exists() else "Missing"
        }

    def _extract_topics(self, conversations: List[Conversation]) -> List[str]:
        all_topics = []
        for conv in conversations:
            all_topics.extend(conv.topics)
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

# --- Rich CLI ---
class RichMultiAgentCLI:
    def __init__(self, api_key: str):
        self.console = Console()
        self.assistant = AutonomousMultiAgentAssistant(api_key)
        self.user_id = "default_user"
        self.conversation_count = 0
        self.session_start = datetime.now()

    def display_banner(self):
        banner_text = """
        ╭─────────────────────────────────────────────────────────────╮
        │                MULTI-AGENT AUTONOMOUS ASSISTANT             │
        │                   Enhanced with Rich CLI                    │
        ╰─────────────────────────────────────────────────────────────╯
        """
        panel = Panel(banner_text, style="bold cyan", border_style="bright_blue", expand=False)
        self.console.print(Align.center(panel))
        status_table = Table(show_header=False, box=None, padding=(0, 2))
        status_table.add_column(style="bold green")
        status_table.add_column(style="cyan")
        status_table.add_row("🤖 Agents:", "Generator → Analyzer → Optimizer")
        status_table.add_row("📁 Data Storage:", "Text-based persistence")
        status_table.add_row("🧠 Context Aware:", "Conversation history integration")
        status_table.add_row("⚡ Autonomous Tasks:", "Background task execution")
        self.console.print(Panel(status_table, title="[bold green]System Features[/bold green]", border_style="green"))

    def display_help(self):
        commands_table = Table(show_header=True, header_style="bold magenta")
        commands_table.add_column("Command", style="bold cyan", width=15)
        commands_table.add_column("Description", style="white")
        commands_table.add_column("Example", style="dim")
        commands = [
            ("chat", "Start conversation with multi-agent system", "How do smart homes work?"),
            ("insights", "View user interaction statistics", ""),
            ("tasks", "Show autonomous tasks and status", ""),
            ("quality", "Display recent quality scores", ""),
            ("agents", "Show multi-agent architecture info", ""),
            ("context", "Preview conversation context", ""),
            ("files", "Check data file status", ""),
            ("stats", "System performance statistics", ""),
            ("clear", "Clear the console screen", ""),
            ("help", "Show this help menu", ""),
            ("docs", "Manage uploaded documents", "upload, list, delete"),
            ("quit", "Exit the application", ""),
        ]
        for cmd, desc, example in commands:
            commands_table.add_row(cmd, desc, example)
        self.console.print(Panel(commands_table, title="[bold magenta]Available Commands[/bold magenta]", border_style="magenta"))

    def show_progress(self, progress, task_id, steps):
        """Show progress animation for multi-agent processing."""
        for complete, description in steps:
            progress.update(task_id, completed=complete, description=description)
            time.sleep(0.5)  # Simulate processing time
    async def process_message_with_progress(self, message: str) -> str:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
        ) as progress:
            agent_task = progress.add_task("Multi-agent processing...", total=100)

            steps = [
                (20, "Generator agent working..."),
                (40, "Generator agent working..."),
                (60, "Analyzer agent reviewing..."),
                (80, "Optimizer agent refining..."),
                (100, "Processing complete!")
            ]

            # Run progress in separate thread
            progress_thread = threading.Thread(
                target=self.show_progress,
                args=(progress, agent_task, steps)
            )
            progress_thread.start()

            response = await self.assistant.process_user_message(self.user_id, message)
            progress_thread.join()

        return response

    def display_response(self, user_message: str, ai_response: str, processing_time: float):
        user_panel = Panel(user_message, title="[bold blue]You[/bold blue]", border_style="blue", padding=(1, 2))
        self.console.print(user_panel)
        conversations = self.assistant.file_manager.get_recent_conversations(self.user_id, 1)
        quality_info = ""
        if conversations and conversations[0].multi_agent_session:
            session = conversations[0].multi_agent_session
            quality_info = f"Quality: {session.quality_score:.2f} | Iterations: {session.total_iterations} | Time: {processing_time:.1f}s"
        ai_panel = Panel(
            Text(ai_response),
            title=f"[bold green]Multi-Agent Assistant[/bold green]",
            subtitle=f"[dim]{quality_info}[/dim]" if quality_info else None,
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(ai_panel)
        self.conversation_count += 1

    def display_insights(self):
        insights = self.assistant.get_user_insights(self.user_id)
        if "message" in insights:
            self.console.print(Panel(insights["message"], title="[yellow]Insights[/yellow]", border_style="yellow"))
            return
        layout = Layout()
        layout.split_column(Layout(name="stats", size=8), Layout(name="details"))
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column(style="bold cyan", width=25)
        stats_table.add_column(style="white")
        stats_table.add_row("Total Conversations:", str(insights.get('total_conversations', 0)))
        stats_table.add_row("Multi-Agent Sessions:", str(insights.get('multi_agent_sessions', 0)))
        stats_table.add_row("Average Quality Score:", f"{insights.get('average_quality_score', 0):.2f}")
        stats_table.add_row("Average Iterations:", f"{insights.get('average_iterations', 0):.1f}")
        stats_table.add_row("Context Usage Rate:", f"{insights.get('context_usage_rate', 0)}%")
        stats_table.add_row("Session Duration:", str(datetime.now() - self.session_start).split('.')[0])
        layout["stats"].update(Panel(stats_table, title="[bold cyan]Session Statistics[/bold cyan]", border_style="cyan"))
        topics = insights.get('topics', [])
        if topics:
            topics_table = Table(show_header=True, header_style="bold magenta")
            topics_table.add_column("Topic", style="cyan")
            topics_table.add_column("Frequency", style="white", justify="right")
            for topic, count in topics[:5]:
                topics_table.add_row(topic, str(count))
        else:
            topics_table = Text("No topics extracted yet", style="dim")
        layout["details"].update(Panel(topics_table, title="[bold magenta]Discussion Topics[/bold magenta]", border_style="magenta"))
        self.console.print(layout)

    def display_tasks(self):
        tasks = self.assistant.file_manager.get_pending_tasks()
        if not tasks:
            self.console.print(Panel("No pending autonomous tasks", title="[yellow]Autonomous Tasks[/yellow]", border_style="yellow"))
            return
        tasks_table = Table(show_header=True, header_style="bold green")
        tasks_table.add_column("ID", style="dim", width=15)
        tasks_table.add_column("Description", style="cyan")
        tasks_table.add_column("Priority", justify="center", width=8)
        tasks_table.add_column("Status", justify="center", width=10)
        tasks_table.add_column("Scheduled", style="dim", width=16)
        for task in tasks[:10]:
            status_style = {"pending": "yellow", "in_progress": "blue", "completed": "green", "failed": "red"}.get(task.status, "white")
            tasks_table.add_row(
                task.id.split('_')[-1],
                task.description[:50] + "..." if len(task.description) > 50 else task.description,
                str(task.priority),
                f"[{status_style}]{task.status}[/{status_style}]",
                task.scheduled_for.strftime("%H:%M:%S")
            )
        self.console.print(Panel(tasks_table, title=f"[bold green]Autonomous Tasks ({len(tasks)} total)[/bold green]", border_style="green"))

    def display_quality_history(self):
        conversations = self.assistant.file_manager.get_recent_conversations(self.user_id, 10)
        if not conversations:
            self.console.print(Panel("No conversation history available", title="[yellow]Quality History[/yellow]", border_style="yellow"))
            return
        quality_table = Table(show_header=True, header_style="bold blue")
        quality_table.add_column("Time", style="dim", width=12)
        quality_table.add_column("Quality Score", justify="center", width=15)
        quality_table.add_column("Iterations", justify="center", width=10)
        quality_table.add_column("Context", justify="center", width=8)
        quality_table.add_column("Preview", style="cyan")
        for conv in conversations:
            if conv.multi_agent_session:
                session = conv.multi_agent_session
                score = session.quality_score
                score_style = "green" if score >= 0.85 else "yellow" if score >= 0.7 else "red"
                score_display = f"[{score_style}]{score:.2f}[/{score_style}]"
                context_indicator = "✓" if len(conv.context) > 100 else "○"
                preview = conv.user_message[:40] + "..." if len(conv.user_message) > 40 else conv.user_message
                quality_table.add_row(conv.timestamp.strftime("%H:%M:%S"), score_display, str(session.total_iterations), context_indicator, preview)
        self.console.print(Panel(quality_table, title="[bold blue]Recent Quality Scores[/bold blue]", border_style="blue"))

    def display_agent_architecture(self):
        tree = Tree("🤖 Multi-Agent System Architecture")
        generator_branch = tree.add("🔧 [bold green]Generator Agent[/bold green]")
        generator_branch.add("• Creates initial solutions")
        generator_branch.add("• Uses conversation history")
        generator_branch.add("• Temperature: 0.8 (creative)")
        analyzer_branch = tree.add("🔍 [bold yellow]Analyzer Agent[/bold yellow]")
        analyzer_branch.add("• Reviews solutions critically")
        analyzer_branch.add("• Identifies errors and gaps")
        analyzer_branch.add("• Temperature: 0.3 (analytical)")
        optimizer_branch = tree.add("⚡ [bold blue]Optimizer Agent[/bold blue]")
        optimizer_branch.add("• Refines and improves")
        optimizer_branch.add("• Maintains continuity")
        optimizer_branch.add("• Temperature: 0.5 (balanced)")
        flow_branch = tree.add("🔄 [bold magenta]Processing Flow[/bold magenta]")
        flow_branch.add("• Max 3 iterations")
        flow_branch.add("• Quality threshold: 0.85")
        flow_branch.add("• Context-aware processing")
        self.console.print(Panel(tree, title="[bold cyan]System Architecture[/bold cyan]", border_style="cyan"))

    def display_context_preview(self):
        context = self.assistant.file_manager.get_conversation_context(self.user_id, 3)
        if not context or context == "No previous conversation history available.":
            self.console.print(Panel("No conversation context available", title="[yellow]Context Preview[/yellow]", border_style="yellow"))
            return
        preview = context[:800] + "\n..." if len(context) > 800 else context
        context_panel = Panel(
            Markdown(preview) if len(preview) < 500 else preview,
            title=f"[bold cyan]Context Preview ({len(context)} chars)[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(context_panel)

    def display_file_status(self):
        file_manager = self.assistant.file_manager
        status_table = Table(show_header=True, header_style="bold green")
        status_table.add_column("File", style="cyan", width=20)
        status_table.add_column("Status", justify="center", width=10)
        status_table.add_column("Size", justify="right", width=12)
        status_table.add_column("Records", justify="right", width=10)
        conv_file = file_manager.conversations_file
        conv_exists = conv_file.exists()
        conv_size = conv_file.stat().st_size if conv_exists else 0
        conv_records = len(file_manager._read_json_file(conv_file)) if conv_exists else 0
        status_table.add_row("conversations.json", "[green]✓[/green]" if conv_exists else "[red]✗[/red]", f"{conv_size:,} bytes", str(conv_records))
        task_file = file_manager.tasks_file
        task_exists = task_file.exists()
        task_size = task_file.stat().st_size if task_exists else 0
        task_records = len(file_manager._read_json_file(task_file)) if task_exists else 0
        status_table.add_row("tasks.json", "[green]✓[/green]" if task_exists else "[red]✗[/red]", f"{task_size:,} bytes", str(task_records))
        user_conversations = len([c for c in file_manager._read_json_file(conv_file) if c.get('user_id') == self.user_id]) if conv_exists else 0
        status_table.add_row(f"Your conversations", "[cyan]ℹ[/cyan]", "-", str(user_conversations))
        self.console.print(Panel(status_table, title=f"[bold green]Data Files Status[/bold green]", subtitle=f"[dim]Data directory: {file_manager.data_dir}[/dim]", border_style="green"))

    def display_system_stats(self):
        stats_layout = Layout()
        stats_layout.split_row(Layout(name="performance"), Layout(name="storage"))
        perf_table = Table(show_header=False, box=None)
        perf_table.add_column(style="bold cyan", width=20)
        perf_table.add_column(style="white")
        uptime = datetime.now() - self.session_start
        perf_table.add_row("Session Uptime:", str(uptime).split('.')[0])
        perf_table.add_row("Conversations:", str(self.conversation_count))
        perf_table.add_row("Assistant Status:", "[green]Active[/green]")
        stats_layout["performance"].update(Panel(perf_table, title="[bold cyan]Performance[/bold cyan]", border_style="cyan"))
        storage_stats = self.assistant.file_manager.get_storage_stats()
        storage_table = Table(show_header=False, box=None)
        storage_table.add_column(style="bold green", width=20)
        storage_table.add_column(style="white")
        storage_table.add_row("Total Conversations:", str(storage_stats.get('total_conversations', 0)))
        storage_table.add_row("Total Tasks:", str(storage_stats.get('total_tasks', 0)))
        storage_table.add_row("Unique Users:", str(len(storage_stats.get('users', {}))))
        stats_layout["storage"].update(Panel(storage_table, title="[bold green]Storage[/bold green]", border_style="green"))
        self.console.print(stats_layout)

    def upload_document(self):
        """Handle document upload process."""
        file_path = Prompt.ask("[bold cyan]Enter file path to upload[/bold cyan]")

        if not file_path.strip():
            self.console.print("[yellow]No file path provided[/yellow]")
            return

        file_path = Path(file_path.strip().strip('"\''))

        if not file_path.exists():
            self.console.print(f"[red]File not found: {file_path}[/red]")
            return

        # Check file size (limit to 5MB for example)
        file_size = file_path.stat().st_size
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            self.console.print("[red]File too large. Maximum size is 5MB[/red]")
            return

        with Status("Processing document...", console=self.console):
            document = self.assistant.file_manager.process_uploaded_file(str(file_path), self.user_id)

        if document:
            self.console.print(f"[green]✓ Document uploaded successfully![/green]")

            # Display document info
            info_table = Table(show_header=False, box=None)
            info_table.add_column(style="bold cyan", width=15)
            info_table.add_column(style="white")
            info_table.add_row("Filename:", document.filename)
            info_table.add_row("Type:", document.file_type)
            info_table.add_row("Size:", decimal(document.file_size))
            info_table.add_row("Content preview:",
                               document.content[:100] + "..." if len(document.content) > 100 else document.content)

            self.console.print(Panel(info_table, title="[bold green]Document Info[/bold green]", border_style="green"))
        else:
            self.console.print("[red]Failed to process document[/red]")

    def display_documents(self):
        """Display user's uploaded documents."""
        documents = self.assistant.file_manager.get_user_documents(self.user_id)

        if not documents:
            self.console.print(
                Panel("No documents uploaded yet", title="[yellow]Your Documents[/yellow]", border_style="yellow"))
            return

        docs_table = Table(show_header=True, header_style="bold cyan")
        docs_table.add_column("ID", style="dim", width=8)
        docs_table.add_column("Filename", style="cyan", width=25)
        docs_table.add_column("Type", style="white", width=15)
        docs_table.add_column("Size", justify="right", width=10)
        docs_table.add_column("Uploaded", style="dim", width=16)
        docs_table.add_column("Preview", style="green")

        for doc in documents:
            preview = doc.content[:50] + "..." if len(doc.content) > 50 else doc.content
            preview = preview.replace('\n', ' ').replace('\r', ' ')

            docs_table.add_row(
                doc.id.split('_')[-1],
                doc.filename,
                doc.file_type.split('/')[-1],
                decimal(doc.file_size),
                doc.upload_time.strftime("%m-%d %H:%M"),
                preview
            )

        self.console.print(Panel(docs_table, title=f"[bold cyan]Your Documents ({len(documents)} total)[/bold cyan]",
                                 border_style="cyan"))

    def manage_documents(self):
        """Document management menu."""
        while True:
            self.console.print()
            choice = Prompt.ask(
                "[bold magenta]Document Management[/bold magenta]",
                choices=["upload", "list", "delete", "back"],
                default="list"
            )

            if choice == "back":
                break
            elif choice == "upload":
                self.upload_document()
            elif choice == "list":
                self.display_documents()
            elif choice == "delete":
                self.delete_document_interactive()

    def delete_document_interactive(self):
        """Interactive document deletion."""
        documents = self.assistant.file_manager.get_user_documents(self.user_id)

        if not documents:
            self.console.print("[yellow]No documents to delete[/yellow]")
            return

        # Show documents with numbers
        self.console.print("\n[bold]Available documents:[/bold]")
        for i, doc in enumerate(documents, 1):
            self.console.print(f"{i}. {doc.filename} ({decimal(doc.file_size)})")

        try:
            choice = Prompt.ask("Enter document number to delete (or 'cancel')")
            if choice.lower() == 'cancel':
                return

            doc_index = int(choice) - 1
            if 0 <= doc_index < len(documents):
                doc_to_delete = documents[doc_index]
                if Confirm.ask(f"Delete '{doc_to_delete.filename}'?"):
                    success = self.assistant.file_manager.delete_document(doc_to_delete.id, self.user_id)
                    if success:
                        self.console.print("[green]✓ Document deleted successfully![/green]")
                    else:
                        self.console.print("[red]Failed to delete document[/red]")
            else:
                self.console.print("[red]Invalid document number[/red]")
        except ValueError:
            self.console.print("[red]Invalid input[/red]")

    async def run_interactive(self):
        self.display_banner()
        self.console.print("\n")
        with Status("Starting multi-agent assistant...", console=self.console):
            self.assistant.start()
            await asyncio.sleep(1)
        self.console.print("[bold green]✓ Multi-Agent Assistant is now running![/bold green]\n")
        self.display_help()
        try:
            while True:
                self.console.print()
                command = Prompt.ask(
                    "[bold blue]Enter command[/bold blue]",
                    choices=["chat", "docs", "insights", "tasks", "quality", "agents", "context", "files", "stats",
                             "clear", "help", "quit"],
                    default="chat"
                )
                if command == "quit":
                    if Confirm.ask("Are you sure you want to quit?"):
                        break
                    continue
                elif command == "chat":
                    message = Prompt.ask("[bold cyan]Your message[/bold cyan]")
                    if message.strip():
                        start_time = time.time()
                        response = await self.process_message_with_progress(message)
                        end_time = time.time()
                        self.display_response(message, response, end_time - start_time)
                elif command == "insights":
                    self.display_insights()
                elif command == "tasks":
                    self.display_tasks()
                elif command == "quality":
                    self.display_quality_history()
                elif command == "agents":
                    self.display_agent_architecture()
                elif command == "docs":
                    self.manage_documents()
                elif command == "context":
                    self.display_context_preview()
                elif command == "files":
                    self.display_file_status()
                elif command == "stats":
                    self.display_system_stats()
                elif command == "clear":
                    self.console.clear()
                    self.display_banner()
                elif command == "help":
                    self.display_help()
        finally:
            with Status("Shutting down multi-agent assistant...", console=self.console):
                self.assistant.stop()
                await asyncio.sleep(1)
            self.console.print("\n[bold green]Multi-Agent Assistant stopped successfully![/bold green]")
            self.console.print("[dim]Thank you for using the Enhanced Multi-Agent Assistant![/dim]")

# --- Main ---
def main():
    console = Console()
    # --- TEMPORARY: Replace with your actual API key ---
    api_key = "AIzaSyCVQuVDmdnoURtxVCZl0ay_Gt5rpBnQOK4"  # <-- Paste your key between the quotes
    # ------------------------------------------------
    if not api_key:
        console.print("[bold red]API key is required to run the assistant.[/bold red]")
        return
    with console.status("Testing Gemini API connection..."):
        try:
            genai.configure(api_key=api_key)
            test_model = genai.GenerativeModel('gemini-1.5-flash')
            test_response = test_model.generate_content("Hello")
        except Exception as e:
            console.print(f"[bold red]API connection failed: {e}[/bold red]")
            return
    console.print("[bold green]✓ API connection successful![/bold green]")
    cli = RichMultiAgentCLI(api_key)
    try:
        asyncio.run(cli.run_interactive())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    import threading
    main()