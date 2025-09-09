import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from dataclasses import dataclass, asdict
import logging
import os
from pathlib import Path
from enum import Enum
import traceback
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"


@dataclass
class AgentResponse:
    agent_role: AgentRole
    content: str
    confidence: float  # 0-1 score
    suggestions: List[str]
    errors_found: List[str]
    improvements: List[str]
    metadata: Dict[str, Any]

    def to_dict(self):  # Add this method
        """Convert AgentResponse to JSON-serializable dictionary"""
        return {
            'agent_role': self.agent_role.value,  # Convert enum to string
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
    context_used: str  # Added to track what context was used

    def to_dict(self):
        # Convert iterations to be JSON serializable
        serializable_iterations = []
        for iteration in self.iterations:
            serializable_iteration = iteration.copy()

            # Convert AgentResponse objects to dictionaries
            for key in ['generator_response', 'analyzer_response', 'optimizer_response']:
                if key in serializable_iteration and hasattr(serializable_iteration[key], 'to_dict'):
                    serializable_iteration[key] = serializable_iteration[key].to_dict()
                elif key in serializable_iteration and isinstance(serializable_iteration[key], dict):
                    # Handle case where it's already a dict from asdict()
                    if 'agent_role' in serializable_iteration[key] and hasattr(
                            serializable_iteration[key]['agent_role'], 'value'):
                        serializable_iteration[key]['agent_role'] = serializable_iteration[key]['agent_role'].value

            serializable_iterations.append(serializable_iteration)

        return {
            'session_id': self.session_id,
            'user_query': self.user_query,
            'iterations': serializable_iterations,
            'final_response': self.final_response,
            'quality_score': self.quality_score,
            'total_iterations': self.total_iterations,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'context_used': self.context_used
        }


@dataclass
class Conversation:
    id: str
    user_id: str
    timestamp: datetime
    user_message: str
    ai_response: str
    multi_agent_session: Optional[MultiAgentSession]
    context: str
    sentiment: str
    topics: List[str]
    potential_followups: List[str]

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
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
            # Handle both old and new format
            context_used = multi_agent_data.get('context_used', '')
            multi_agent_session = MultiAgentSession(
                session_id=multi_agent_data['session_id'],
                user_query=multi_agent_data['user_query'],
                iterations=multi_agent_data['iterations'],
                final_response=multi_agent_data['final_response'],
                quality_score=multi_agent_data['quality_score'],
                total_iterations=multi_agent_data['total_iterations'],
                timestamp=datetime.fromisoformat(multi_agent_data['timestamp']),
                context_used=context_used
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

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            id=data['id'],
            conversation_id=data['conversation_id'],
            task_type=data['task_type'],
            description=data['description'],
            priority=data['priority'],
            status=data['status'],
            created_at=datetime.fromisoformat(data['created_at']),
            scheduled_for=datetime.fromisoformat(data['scheduled_for']),
            result=data.get('result'),
            metadata=data.get('metadata')
        )


class TextFileManager:
    def __init__(self, data_dir: str = "multi_agent_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.conversations_file = self.data_dir / "conversations.txt"
        self.tasks_file = self.data_dir / "tasks.txt"
        self.agent_sessions_file = self.data_dir / "agent_sessions.txt"

        self._ensure_files_exist()

    def _ensure_files_exist(self):
        for file_path in [self.conversations_file, self.tasks_file, self.agent_sessions_file]:
            if not file_path.exists():
                file_path.write_text("[]")

    def _read_json_file(self, file_path: Path) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.info(f"Empty file {file_path}, initializing with empty list")
                    return []
                data = json.loads(content)
                logger.info(f"Successfully loaded {len(data)} records from {file_path}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            # Backup corrupted file
            backup_path = file_path.with_suffix('.backup')
            file_path.rename(backup_path)
            logger.info(f"Corrupted file backed up to {backup_path}")
            return []
        except FileNotFoundError:
            logger.info(f"File {file_path} not found, will be created on first save")
            return []

    def _write_json_file(self, file_path: Path, data: List[Dict]):
        """Write JSON file with atomic operations and validation."""
        import tempfile
        import shutil

        try:
            # Validate data can be serialized
            json_content = json.dumps(data, indent=2, ensure_ascii=False)

            # Write to temporary file first (atomic operation)
            temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=file_path.parent)

            try:
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(json_content)
                    temp_file.flush()
                    os.fsync(temp_fd)  # Force write to disk

                # Atomic replace - this prevents corruption
                shutil.move(temp_path, file_path)
                logger.info(f"Successfully wrote {len(data)} records to {file_path}")

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            raise

    def _sanitize_for_json(self, text: str) -> str:
        """Remove problematic characters that break JSON."""
        if not isinstance(text, str):
            return str(text)

        # Remove control characters except newline, carriage return, tab
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Escape problematic characters
        sanitized = sanitized.replace('\\', '\\\\').replace('"', '\\"')

        return sanitized

    def save_conversation(self, conversation: Conversation):
        """Save conversation with better error handling and validation."""
        try:
            print(f" DEBUG: Attempting to save conversation {conversation.id}")

            # Read existing conversations
            conversations = self._read_json_file(self.conversations_file)
            print(f" DEBUG: Loaded {len(conversations)} existing conversations")

            # Remove any existing conversation with same ID
            conversations = [c for c in conversations if c.get('id') != conversation.id]

            # Convert to dict with proper serialization
            conv_dict = conversation.to_dict()

            # Sanitize problematic content
            conv_dict['user_message'] = self._sanitize_for_json(conv_dict['user_message'])
            conv_dict['ai_response'] = self._sanitize_for_json(conv_dict['ai_response'])
            conv_dict['context'] = self._sanitize_for_json(conv_dict['context'])

            # Add the new conversation
            conversations.append(conv_dict)

            # Sort by timestamp (newest first)
            conversations = sorted(conversations,
                                   key=lambda x: x.get('timestamp', ''),
                                   reverse=True)

            # Write to file
            self._write_json_file(self.conversations_file, conversations)

            print(f" DEBUG: Successfully saved conversation {conversation.id}")
            print(f" DEBUG: Total conversations in file: {len(conversations)}")

        except Exception as e:
            print(f" DEBUG: Failed to save conversation: {e}")
            print(f" DEBUG: Exception details: {traceback.format_exc()}")
            logger.error(f"Failed to save conversation: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    def save_task(self, task: Task):
        """Save task with better error handling."""
        try:
            tasks = self._read_json_file(self.tasks_file)

            # Remove existing task with same ID
            tasks = [t for t in tasks if t.get('id') != task.id]

            # Add new task
            task_dict = task.to_dict()
            tasks.append(task_dict)

            # Sort by created_at (newest first) but keep all permanently
            tasks = sorted(tasks, key=lambda x: x.get('created_at', ''), reverse=True)

            self._write_json_file(self.tasks_file, tasks)
            logger.info(f"Successfully saved task {task.id}")

        except Exception as e:
            logger.error(f"Failed to save task: {e}")
            print(f" Error saving task: {e}")

    def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Conversation]:
        conversations_data = self._read_json_file(self.conversations_file)
        user_conversations = [c for c in conversations_data if c.get('user_id') == user_id]
        user_conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        conversations = []
        for conv_data in user_conversations[:limit]:
            try:
                conversations.append(Conversation.from_dict(conv_data))
            except Exception as e:
                logger.warning(f"Skipping malformed conversation data: {e}")

        logger.info(f" Retrieved {len(conversations)} conversations from {self.conversations_file}")
        return conversations

    def get_conversation_context(self, user_id: str, limit: int = 15) -> str:
        """Get formatted conversation context for agents with debugging."""
        conversations = self.get_recent_conversations(user_id, limit)

        print(f" DEBUG: Found {len(conversations)} conversations for context")

        if not conversations:
            print(" DEBUG: No conversation history found!")
            return "No previous conversation history available."

        context_parts = ["=== PREVIOUS CONVERSATION HISTORY ==="]
        for i, conv in enumerate(conversations):
            print(f" DEBUG: Adding conversation {i + 1}: {conv.user_message[:50]}...")
            context_parts.append(f"\n--- Conversation {i + 1} ({conv.timestamp.strftime('%Y-%m-%d %H:%M')}) ---")
            context_parts.append(f"USER: {conv.user_message}")
            context_parts.append(f"ASSISTANT: {conv.ai_response}")

            if conv.multi_agent_session:
                context_parts.append(
                    f"[Quality: {conv.multi_agent_session.quality_score:.2f}, Iterations: {conv.multi_agent_session.total_iterations}]")

            if conv.topics:
                context_parts.append(f"[Topics: {', '.join(conv.topics)}]")

        context_parts.append("\n=== END OF CONVERSATION HISTORY ===")
        full_context = "\n".join(context_parts)

        print(f" DEBUG: Built context with {len(full_context)} characters")
        return full_context

    def get_pending_tasks(self) -> List[Task]:
        tasks_data = self._read_json_file(self.tasks_file)
        now = datetime.now()

        pending_tasks = []
        for task_data in tasks_data:
            try:
                if (task_data.get('status') == 'pending' and
                        datetime.fromisoformat(task_data.get('scheduled_for', '')) <= now):
                    pending_tasks.append(Task.from_dict(task_data))
            except Exception as e:
                logger.warning(f"Skipping malformed task data: {e}")

        pending_tasks.sort(key=lambda x: (-x.priority, x.scheduled_for))
        return pending_tasks

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get detailed storage statistics."""
        conversations_data = self._read_json_file(self.conversations_file)
        tasks_data = self._read_json_file(self.tasks_file)

        user_stats = {}
        for conv in conversations_data:
            user_id = conv.get('user_id', 'unknown')
            if user_id not in user_stats:
                user_stats[user_id] = 0
            user_stats[user_id] += 1

        return {
            'total_conversations': len(conversations_data),
            'total_tasks': len(tasks_data),
            'users': user_stats,
            'oldest_conversation': min([c.get('timestamp', '') for c in conversations_data], default='none'),
            'newest_conversation': max([c.get('timestamp', '') for c in conversations_data], default='none'),
            'files_exist': {
                'conversations': self.conversations_file.exists(),
                'tasks': self.tasks_file.exists()
            }
        }

    def debug_file_status(self):
        """Debug method to check file status and contents"""
        print(f"\n DEBUG: File System Status")
        print(f"Data directory: {self.data_dir}")
        print(f"Data directory exists: {self.data_dir.exists()}")
        print(f"Conversations file: {self.conversations_file}")
        print(f"Conversations file exists: {self.conversations_file.exists()}")

        if self.conversations_file.exists():
            try:
                file_size = self.conversations_file.stat().st_size
                print(f"File size: {file_size} bytes")

                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        data = json.loads(content)
                        print(f"Conversations in file: {len(data)}")
                    else:
                        print("File is empty")
            except Exception as e:
                print(f"Error reading file: {e}")


class AIAgent:
    def __init__(self, api_key: str, role: AgentRole, temperature: float = 0.7):
        genai.configure(api_key=api_key)
        self.role = role
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
        )

    async def process(self, user_query: str, context: str = "", previous_iteration: Dict = None) -> AgentResponse:
        """Process based on agent role with improved context usage."""
        logger.info(f" {self.role.value} agent processing with context length: {len(context)}")

        if self.role == AgentRole.GENERATOR:
            return await self._generate_solution(user_query, context)
        elif self.role == AgentRole.ANALYZER:
            return await self._analyze_solution(user_query, context, previous_iteration)
        elif self.role == AgentRole.OPTIMIZER:
            return await self._optimize_solution(user_query, context, previous_iteration)

    async def _generate_solution(self, user_query: str, context: str) -> AgentResponse:
        """Agent 1: Generate initial solution using conversation history."""
        prompt = f"""
        Role: Solution Generator Agent
        Task: Provide a comprehensive initial solution to the user's query. Use available context naturally without mentioning it.

        {context}

        Current User Query: {user_query}

        Instructions:
        1. Use the conversation history to inform your response but do not mention it explicitly
        2. Provide a detailed, well-structured response that naturally incorporates relevant context
        3. Do not reference previous conversations or mention using conversation history
        4. Make the response feel complete and self-contained
        5. Include practical examples where relevant
        6. Consider multiple approaches if applicable
        7. Be thorough but clear
        8. Rate your confidence in this solution (0-1)

        Format your response as JSON:
        {{
            "solution": "Your detailed solution here (incorporate context naturally without explicit references)",
            "confidence": 0.85,
            "approaches_considered": ["approach1", "approach2"],
            "examples_included": ["example1", "example2"],
            "assumptions": ["assumption1", "assumption2"]
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
                content=data.get("solution", response_text),
                confidence=data.get("confidence", 0.7),
                suggestions=data.get("approaches_considered", []),
                errors_found=[],
                improvements=[],
                metadata=data
            )
        except Exception as e:
            logger.error(f"Generator agent error: {e}")
            return AgentResponse(
                agent_role=self.role,
                content=f"I can help you with: {user_query}. Let me provide a comprehensive response based on the available information.",
                confidence=0.6,
                suggestions=[],
                errors_found=[],
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
        """Agent 3: Optimize solution using conversation history and analysis."""
        generator_response = previous_iteration.get("generator_response", {})
        analyzer_response = previous_iteration.get("analyzer_response", {})

        original_solution = generator_response.get("content", "")
        analysis = analyzer_response.get("content", "")
        errors = analyzer_response.get("errors_found", [])
        improvements = analyzer_response.get("improvements", [])

        prompt = f"""
        Role: Solution Optimizer Agent
        Task: Create an improved solution based on analysis feedback. Use conversation history for context but do not explicitly mention previous conversations.

        {context}

        Original User Query: {user_query}

        Original Solution:
        {original_solution}

        Analysis Feedback:
        {analysis}

        Identified Errors: {errors}
        Suggested Improvements: {improvements}

        Instructions:
        1. Use the conversation history to inform your response but do not mention it explicitly
        2. Fix all identified errors
        3. Address the gaps and improvements
        4. Enhance clarity and completeness
        5. Provide a natural, direct response that seamlessly incorporates relevant context
        6. Do not reference previous conversations or mention context usage
        7. Make the response feel like a fresh, complete answer
        8. Maintain a helpful tone without meta-commentary about conversation history

        Format your response as JSON:
        {{
            "optimized_solution": "Your improved solution here (incorporate context naturally without mentioning it)",
            "changes_made": ["change1", "change2"],
            "errors_fixed": ["fix1", "fix2"],
            "enhancements": ["enhancement1", "enhancement2"],
            "confidence": 0.95
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
                content=original_solution,  # Fallback to original
                confidence=0.7,
                suggestions=[],
                errors_found=[],
                improvements=[],
                metadata={"error": str(e)}
            )


class MultiAgentSystem:
    def __init__(self, api_key: str):
        self.generator = AIAgent(api_key, AgentRole.GENERATOR, temperature=0.8)
        self.analyzer = AIAgent(api_key, AgentRole.ANALYZER, temperature=0.3)
        self.optimizer = AIAgent(api_key, AgentRole.OPTIMIZER, temperature=0.5)
        self.max_iterations = 3
        self.quality_threshold = 0.85

    async def process_query(self, user_query: str, context: str = "") -> MultiAgentSession:
        """Process query through multi-agent system with agent conversation display."""
        session_id = f"session_{int(time.time())}"
        iterations = []

        current_iteration = 0
        final_response = ""
        quality_score = 0.0

        while current_iteration < self.max_iterations:
            print(f"\n--- Iteration {current_iteration + 1} ---")

            iteration_data = {
                "iteration": current_iteration + 1,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(context)
            }

            # Step 1: Generator
            print(f"🤖 Generator Agent: Creating initial response...")
            generator_response = await self.generator.process(
                user_query,
                context,
                iterations[-1] if iterations else None
            )
            iteration_data["generator_response"] = asdict(generator_response)

            # Show Generator's work
            print(f"📝 Generator says: {generator_response.content[:200]}...")
            print(f"   Confidence: {generator_response.confidence:.2f}")

            # Step 2: Analyzer
            print(f"\n🔍 Analyzer Agent: Reviewing and checking for improvements...")
            analyzer_response = await self.analyzer.process(
                user_query,
                context,
                iteration_data
            )
            iteration_data["analyzer_response"] = asdict(analyzer_response)

            # Show Analyzer's feedback
            print(f"🔍 Analyzer says: {analyzer_response.content[:200]}...")
            if analyzer_response.errors_found:
                print(f"   ❌ Errors found: {', '.join(analyzer_response.errors_found[:2])}")
            if analyzer_response.improvements:
                print(f"   💡 Improvements needed: {', '.join(analyzer_response.improvements[:2])}")
            print(f"   Quality assessment: {analyzer_response.confidence:.2f}")

            # Step 3: Optimizer
            print(f"\n⚡ Optimizer Agent: Creating improved response...")
            optimizer_response = await self.optimizer.process(
                user_query,
                context,
                iteration_data
            )
            iteration_data["optimizer_response"] = asdict(optimizer_response)

            # Show Optimizer's improvements
            print(f"⚡ Optimizer says: {optimizer_response.content[:200]}...")
            if hasattr(optimizer_response, 'improvements') and optimizer_response.improvements:
                print(f"   ✅ Changes made: {', '.join(optimizer_response.improvements[:2])}")
            print(f"   Final confidence: {optimizer_response.confidence:.2f}")

            iterations.append(iteration_data)
            final_response = optimizer_response.content
            quality_score = optimizer_response.confidence

            # Check if we've reached quality threshold
            if quality_score >= self.quality_threshold:
                print(f"\n✅ Quality threshold reached! ({quality_score:.2f} >= {self.quality_threshold})")
                break

            # If we found significant errors, continue iterating
            errors_found = analyzer_response.errors_found
            if not errors_found and current_iteration > 0:
                print(f"\n✅ No significant errors found, finalizing response")
                break

            if current_iteration < self.max_iterations - 1:
                print(f"\n🔄 Quality not sufficient ({quality_score:.2f}), continuing to next iteration...")

            current_iteration += 1

        session = MultiAgentSession(
            session_id=session_id,
            user_query=user_query,
            iterations=iterations,
            final_response=final_response,
            quality_score=quality_score,
            total_iterations=len(iterations),
            timestamp=datetime.now(),
            context_used=context[:500] + "..." if len(context) > 500 else context
        )

        print(f"\n🎯 Multi-agent processing complete!")
        print(f"   Total iterations: {len(iterations)}")
        print(f"   Final quality score: {quality_score:.2f}")

        return session


class AutonomousMultiAgentAssistant:
    def __init__(self, gemini_api_key: str):
        self.file_manager = TextFileManager()
        self.multi_agent_system = MultiAgentSystem(gemini_api_key)
        self.running = False
        self.task_executor_thread = None

    def start(self):
        logger.info(" Starting Multi-Agent Autonomous Assistant...")
        self.running = True
        self.task_executor_thread = threading.Thread(target=self._task_executor_loop, daemon=True)
        self.task_executor_thread.start()
        logger.info(" Multi-Agent Assistant started successfully!")

    def stop(self):
        logger.info(" Stopping Multi-Agent Assistant...")
        self.running = False
        if self.task_executor_thread:
            self.task_executor_thread.join(timeout=5)
        logger.info(" Multi-Agent Assistant stopped.")

    async def process_user_message(self, user_id: str, message: str) -> str:
        """Process user message through multi-agent system with clean output."""

        # Get conversation context (silently)
        context = self.file_manager.get_conversation_context(user_id, 15)

        print("📚 Loading conversation history...")
        print("🚀 Starting multi-agent processing...\n")

        # Process through multi-agent system
        multi_agent_session = await self.multi_agent_system.process_query(message, context)

        # Create and save conversation (silently)
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

        # Save conversation and create tasks (silently)
        try:
            self.file_manager.save_conversation(conversation)
            await self._analyze_and_create_tasks(conversation)
        except Exception as e:
            pass  # Handle errors silently

        return multi_agent_session.final_response

    async def _analyze_and_create_tasks(self, conversation: Conversation):
        """Analyze conversation and create autonomous tasks with better error handling."""
        print(f" Analyzing conversation for task creation...")

        try:
            # Simplified task creation - bypass complex AI analysis for now
            suggested_tasks = []

            # Create basic follow-up tasks based on conversation content
            if "smart home" in conversation.user_message.lower():
                suggested_tasks = [
                    "Research latest smart thermostat models and energy savings",
                    "Compare smart lighting systems within $500-800 budget range"
                ]
            elif "sensors" in conversation.user_message.lower():
                suggested_tasks = [
                    "Research motion sensors for smart lighting automation",
                    "Compare ambient light sensors for energy efficiency"
                ]

            print(f" Generated {len(suggested_tasks)} tasks")

            # Create and save tasks
            for i, task_desc in enumerate(suggested_tasks):
                task_id = f"{conversation.id}_task_{i}"
                schedule_delay = timedelta(minutes=2 + (i * 3))  # Faster scheduling for testing

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
                print(f" Created task: {task_desc}")

        except Exception as e:
            print(f" Error creating tasks: {e}")
            logger.error(f"Task creation error: {e}")

    def _task_executor_loop(self):
        """Execute autonomous tasks."""
        logger.info(" Starting autonomous task executor...")

        while self.running:
            try:
                pending_tasks = self.file_manager.get_pending_tasks()

                for task in pending_tasks[:2]:  # Process 2 tasks at a time
                    asyncio.run(self._execute_task_async(task))

                time.sleep(45)  # Check every 45 seconds

            except Exception as e:
                logger.error(f"Task executor error: {e}")
                time.sleep(60)

    async def _execute_task_async(self, task: Task):
        """Execute task using multi-agent system."""
        logger.info(f" Executing multi-agent task: {task.description}")

        try:
            task.status = "in_progress"
            self.file_manager.save_task(task)

            # Get context for task execution if user_id is available
            context = ""
            if task.metadata and 'user_id' in task.metadata:
                context = self.file_manager.get_conversation_context(task.metadata['user_id'], 3)

            # Use multi-agent system for task execution
            session = await self.multi_agent_system.process_query(
                f"Research and provide information about: {task.description}",
                context
            )

            task.result = session.final_response
            task.status = "completed"
            task.metadata["quality_score"] = session.quality_score
            task.metadata["iterations"] = session.total_iterations

            self.file_manager.save_task(task)
            logger.info(f" Task completed with quality {session.quality_score:.2f}")

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            self.file_manager.save_task(task)

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get user insights including multi-agent statistics."""
        conversations = self.file_manager.get_recent_conversations(user_id, 20)

        if not conversations:
            return {"message": "No conversation history found"}

        # Collect multi-agent statistics
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
            "context_usage_rate": round(context_usage_rate * 100, 1),  # Percentage
            "topics": self._extract_topics(conversations),
            "last_interaction": conversations[0].timestamp.isoformat() if conversations else None,
            "conversation_file_status": " Active" if self.file_manager.conversations_file.exists() else "❌ Missing"
        }

    def _extract_topics(self, conversations: List[Conversation]) -> List[str]:
        """Extract topics from conversations."""
        all_topics = []
        for conv in conversations:
            all_topics.extend(conv.topics)

        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]


class MultiAgentCLI:
    def __init__(self, api_key: str):
        self.assistant = AutonomousMultiAgentAssistant(api_key)
        self.user_id = "default_user"

    async def run_interactive(self):
        """Run interactive multi-agent chat with proper context usage."""
        print(" Enhanced Multi-Agent Autonomous Assistant Started!")
        print(" Features: Context-Aware 3-Agent Processing (Generator → Analyzer → Optimizer)")
        print(" Now properly uses conversation history from .txt files!")
        print("Commands: 'quit', 'insights', 'tasks', 'agents', 'context', 'verbose'\n")
        print("💡 Agent conversations are shown by default during processing!\n")

        self.assistant.start()

        try:
            while True:
                user_input = input("\n You: ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'insights':
                    insights = self.assistant.get_user_insights(self.user_id)
                    print(f"\n Enhanced Multi-Agent Insights:")
                    print(f"   Total Conversations: {insights.get('total_conversations', 0)}")
                    print(f"   Multi-Agent Sessions: {insights.get('multi_agent_sessions', 0)}")
                    print(f"   Average Quality Score: {insights.get('average_quality_score', 0)}")
                    print(f"   Context Usage Rate: {insights.get('context_usage_rate', 0)}%")
                    print(f"   File Status: {insights.get('conversation_file_status', 'Unknown')}")
                    continue
                elif user_input.lower() == 'tasks':
                    tasks = self.assistant.file_manager.get_pending_tasks()
                    print(f"\n⚡ Autonomous Tasks ({len(tasks)} pending):")
                    for task in tasks[:3]:
                        print(f"  - {task.description}")
                        if task.metadata and 'quality_score' in task.metadata:
                            print(f"    Quality: {task.metadata['quality_score']:.2f}")
                    continue
                elif user_input.lower() == 'quality':
                    conversations = self.assistant.file_manager.get_recent_conversations(self.user_id, 5)
                    print(f"\n Recent Quality Scores:")
                    for conv in conversations:
                        if conv.multi_agent_session:
                            context_used = " " if len(conv.context) > 100 else " "
                            print(
                                f"  {conv.timestamp.strftime('%H:%M')}: {conv.multi_agent_session.quality_score:.2f} ({conv.multi_agent_session.total_iterations} iterations) {context_used}")
                    continue
                elif user_input.lower() == 'agents':
                    print("\nEnhanced Multi-Agent System Architecture:")
                    print("  Agent 1 (Generator): Creates solutions using conversation history")
                    print("  Agent 2 (Analyzer): Reviews considering previous context")
                    print("  Agent 3 (Optimizer): Refines while maintaining continuity")
                    print("   Iterates until quality threshold (0.85) or max iterations (3)")
                    print("   Now properly integrates conversation history from .txt files!")
                    continue
                elif user_input.lower() == 'context':
                    context = self.assistant.file_manager.get_conversation_context(self.user_id, 3)
                    print(f"\n Current Context Preview ({len(context)} chars):")
                    print(context[:500] + "..." if len(context) > 500 else context)
                    continue
                elif user_input.lower() == 'files':
                    data_dir = self.assistant.file_manager.data_dir
                    conversations_file = self.assistant.file_manager.conversations_file
                    tasks_file = self.assistant.file_manager.tasks_file
                    continue
                elif user_input.lower() == 'verbose':
                    print("\n🔧 Verbose mode is now enabled by default!")
                    print("You'll see all agent conversations during processing.")
                    print("Commands available:")
                    print("  - 'insights': View conversation statistics")
                    print("  - 'tasks': View autonomous tasks")
                    print("  - 'agents': View agent architecture")
                    print("  - 'context': Preview conversation context")
                    continue

                    print(f"\n File Status:")
                    print(f"  Data Directory: {data_dir}")
                    print(f"  Conversations File: {'correct ' if conversations_file.exists() else 'false '} {conversations_file}")
                    print(f"  Tasks File: {'correct' if tasks_file.exists() else ' false '} {tasks_file}")

                    if conversations_file.exists():
                        conversations_data = self.assistant.file_manager._read_json_file(conversations_file)
                        user_conversations = [c for c in conversations_data if c.get('user_id') == self.user_id]
                        print(f"  Your Conversations: {len(user_conversations)} stored")
                    continue

                if user_input:
                    start_time = time.time()
                    response = await self.assistant.process_user_message(self.user_id, user_input)
                    end_time = time.time()

                    print(
                        f"\n Final Response (Quality Score: {getattr(response, 'quality_score', 'N/A')}, {end_time - start_time:.1f}s):")
                    print(f"\n{response}")

        finally:
            self.assistant.stop()
            print("\n Enhanced Multi-Agent Assistant stopped!")

    def debug_file_contents(self):
        """Debug method to check file contents."""
        conversations_file = self.assistant.file_manager.conversations_file
        if conversations_file.exists():
            try:
                with open(conversations_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f" File size: {len(content)} characters")
                    if content.strip():
                        data = json.loads(content)
                        print(f" Conversations stored: {len(data)}")
                        user_convs = [c for c in data if c.get('user_id') == self.user_id]
                        print(f" Your conversations: {len(user_convs)}")
                    else:
                        print(" File is empty")
            except Exception as e:
                print(f" Error reading file: {e}")
        else:
            print(" Conversations file doesn't exist")


def main():
    """Main function to run the enhanced multi-agent assistant."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Please enter your Gemini API key: ").strip()
        convo_limit=int(input("Enter the limit to the conversational window according to you api key plan:"))

    if not api_key:
        print(" API key is required.")
        return

    print(" Testing Gemini API connection...")
    try:
        genai.configure(api_key=api_key)
        test_model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = test_model.generate_content("Hello")
        print(f" API connection successful!")
        print(f" Using model: gemini-1.5-flash")
        print(f" Enhanced with conversation history integration!")
    except Exception as e:
        print(f" API connection failed: {e}")
        return

    cli = MultiAgentCLI(api_key)

    # Debug file status before starting
    print(f"\n Checking data files...")
    cli.debug_file_contents()
    # Test file system before starting
    print(" Testing file system...")
    test_assistant = AutonomousMultiAgentAssistant(api_key)
    test_assistant.file_manager.debug_file_status()

    # Test if we can write to files
    try:
        test_dir = Path("multi_agent_data")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test_write.txt"
        test_file.write_text("test")
        test_file.unlink()
        print(" File system write test passed")
    except Exception as e:
        print(f"File system write test failed: {e}")
        return

    asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()