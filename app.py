#!/usr/bin/env python
# coding: utf-8
"""Study Guide Generator from Academic Documents - Streamlit Interface"""

import os
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import logging
import hashlib
import json
from datetime import datetime
#######################################################
from pathlib import Path

# Pick a cache folder inside your app that you control
cache_dir = Path(os.getcwd()) / "tiktoken_cache"
cache_dir.mkdir(exist_ok=True)

# Tell tiktoken to use it (must come before any tiktoken import)
os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
#######################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("BlogAssistant")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool
from typing import Union, List
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)
from llama_index.core.agent.workflow import FunctionAgent

# Configure page
st.set_page_config(
    page_title="Study Guide Generator",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "progress_updates" not in st.session_state:
    st.session_state.progress_updates = []
if "study_guide" not in st.session_state:
    st.session_state.study_guide = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "storage_dir" not in st.session_state:
    st.session_state.storage_dir = None

# Configure models via .env
def configure_models():
    # Settings.llm = NVIDIA(model=os.getenv("LLAMA_MODEL", "meta/llama-3.3-70b-instruct"))
    Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    Settings.embed_model = NVIDIAEmbedding(
        model=os.getenv("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
        truncate="END"
    )

# Generate a unique hash for a file to track vector store
def get_file_hash(file_path):
    """Generate a unique hash for a file based on its path and modification time"""
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(file_info.encode()).hexdigest()

# Vector store registry to track what documents have been indexed
class VectorStoreRegistry:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.join(tempfile.gettempdir(), "vector_store_registry")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry_file = os.path.join(self.base_dir, "registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Creating new one.")
                return {}
        return {}
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_document(self, file_path, file_hash, storage_dir):
        """Register a document with its vector store location"""
        self.registry[file_hash] = {
            "file_path": file_path,
            "storage_dir": storage_dir,
            "created_at": datetime.now().isoformat()
        }
        self._save_registry()
    
    def get_storage_dir(self, file_hash):
        """Get the storage directory for a document hash if it exists"""
        if file_hash in self.registry:
            storage_dir = self.registry[file_hash]["storage_dir"]
            if os.path.exists(storage_dir):
                return storage_dir
        return None

# Initialize vector store registry
vector_store_registry = VectorStoreRegistry()

# Load or build index
def get_index(file_path, storage_dir):
    """Get index for document - reuses existing vector store if available"""
    # Generate a hash for this file
    file_hash = get_file_hash(file_path)
    
    # Check if we already have a vector store for this document
    existing_storage_dir = vector_store_registry.get_storage_dir(file_hash)
    
    if existing_storage_dir:
        logger.info(f"✅ Using existing vector store for document {os.path.basename(file_path)}")
        logger.info(f"🔍 RAG: Loading existing vector index from {existing_storage_dir}")
        st.success(f"Using existing vector index for {os.path.basename(file_path)}")
        
        # Add more prominent UI notification
        st.info(f"⚡ PERFORMANCE BOOST: Using pre-computed vector store from previous runs instead of creating a new one. This will make processing faster!")
        
        # Track vector store reuse in session state for RAG Info tab
        if "rag_stats" not in st.session_state:
            st.session_state.rag_stats = {}
        st.session_state.rag_stats["reused_vector_store"] = True
        st.session_state.rag_stats["vector_store_path"] = existing_storage_dir
        
        try:
            ctx = StorageContext.from_defaults(persist_dir=existing_storage_dir)
            index = load_index_from_storage(ctx)
            logger.info(f"🔍 RAG: Vector index loaded successfully with {len(index.docstore.docs)} documents")
            return index
        except Exception as e:
            logger.warning(f"Failed to load existing index, creating new one: {e}")
    
    # Check if the storage directory exists AND contains necessary index files
    docstore_path = os.path.join(storage_dir, "docstore.json")
    
    if os.path.exists(storage_dir) and os.path.exists(docstore_path):
        try:
            logger.info(f"Loading index from {storage_dir}")
            logger.info(f"🔍 RAG: Loading vector index from {storage_dir}")
            ctx = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(ctx)
            
            # Register this storage location for future use
            vector_store_registry.register_document(file_path, file_hash, storage_dir)
            logger.info(f"🔍 RAG: Vector index loaded successfully with {len(index.docstore.docs)} documents")
            
            return index
        except (FileNotFoundError, ValueError, Exception) as e:
            logger.warning(f"Could not load existing index, creating new one: {str(e)}")
            # Fall through to create new index
    
    # Create the storage directory if it doesn't exist
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create new index
    try:
        logger.info(f"📄 Creating new vector index for {os.path.basename(file_path)}")
        logger.info(f"🔍 RAG: Initializing new vector store (this is where embeddings are created)")
        st.info(f"Building new vector index for {os.path.basename(file_path)}. This may take a moment...")
        
        start_time = datetime.now()
        logger.info(f"🔍 RAG: Parsing document content with LlamaParse")
        docs = LlamaParse(result_type="markdown").load_data(file_path)
        
        loading_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Document parsed in {loading_time:.2f} seconds")
        logger.info(f"🔍 RAG: Document parsed into {len(docs)} chunks")
        
        # Log chunk details for transparency
        for i, doc in enumerate(docs[:3]):  # Show first 3 chunks only
            content_preview = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
            logger.info(f"🔍 RAG: Chunk {i+1} preview: {content_preview}")
        
        if len(docs) > 3:
            logger.info(f"🔍 RAG: ... and {len(docs)-3} more chunks")
        
        logger.info(f"🔍 RAG: Creating embeddings and vector store from {len(docs)} chunks")
        idx = VectorStoreIndex.from_documents(docs)
        
        nodes_count = len(idx.docstore.docs)
        logger.info(f"🔍 RAG: Vector store created with {nodes_count} nodes")
        
        logger.info(f"🔍 RAG: Persisting vector store to {storage_dir}")
        idx.storage_context.persist(persist_dir=storage_dir)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Vector index created in {total_time:.2f} seconds")
        logger.info(f"🔍 RAG: Vector store ready for retrieval with {nodes_count} embeddings")
        
        # Register this storage location for future use
        vector_store_registry.register_document(file_path, file_hash, storage_dir)
        
        # Track that we created a new vector store in session state
        if "rag_stats" not in st.session_state:
            st.session_state.rag_stats = {}
        st.session_state.rag_stats["reused_vector_store"] = False
        st.session_state.rag_stats["new_vector_store"] = True
        st.session_state.rag_stats["vector_store_path"] = storage_dir
        st.session_state.rag_stats["chunks_count"] = nodes_count
        
        return idx
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")

# Wrap query engine as tool
def make_document_tool(index, name, description):
    engine = index.as_query_engine(similarity_top_k=10)
    return QueryEngineTool.from_defaults(
        engine,
        name=name,
        description=description,
    )

# Event classes
class OutlineEvent(Event):
    outline: str

class QuestionEvent(Event):
    question: str

class AnswerEvent(Event):
    question: str
    answer: str

class ReviewEvent(Event):
    study_guide: str

class ProgressEvent(Event):
    progress: str
    step: str = ""  # Add step information

# Workflow definition
class DocumentResearchAgent(Workflow):
    @step
    async def formulate_plan(self, ctx: Context, ev: StartEvent) -> OutlineEvent:
        logger.info(f"Starting workflow step: formulate_plan")
        query = ev.query
        await ctx.set("original_query", query)
        await ctx.set("tools", ev.tools)
        prompt = f"Plan a detailed outline (3 bullet points) for: {query}"
        ctx.write_event_to_stream(ProgressEvent(progress="Working on outline...", step="Creating Outline"))
        logger.info(f"Generating outline for query: {query}")
        resp = await Settings.llm.acomplete(prompt)
        ctx.write_event_to_stream(ProgressEvent(progress="Outline completed", step="Outline Created"))
        logger.info(f"Completed step formulate_plan → OutlineEvent")
        return OutlineEvent(outline=str(resp))

    @step
    async def formulate_questions(self, ctx: Context, ev: OutlineEvent) -> None:
        logger.info(f"🔍 STEP 1: formulate_questions - Generating research questions")
        outline = ev.outline
        await ctx.set("outline", outline)
        prompt = f"Generate 2 concise questions for outline: {outline}"
        ctx.write_event_to_stream(ProgressEvent(progress="Generating research questions...", step="Creating Questions"))
        logger.info("Generating research questions based on outline")
        resp = await Settings.llm.acomplete(prompt)
        questions = [q for q in str(resp).splitlines() if q and not q.lower().startswith('here')]
        questions = questions[:2]
        await ctx.set("num_questions", len(questions))
        for i, q in enumerate(questions):
            logger.info(f"Question {i+1}: {q[:50]}...")
            ctx.write_event_to_stream(ProgressEvent(
                progress=f"Generated question {i+1}/{len(questions)}", 
                step="Question Created"
            ))
            ctx.send_event(QuestionEvent(question=q))
        logger.info(f"🔍 STEP 1 COMPLETE: formulate_questions - Generated {len(questions)} questions")

    @step
    async def answer_question(self, ctx: Context, ev: QuestionEvent) -> AnswerEvent:
        logger.info(f"📚 STEP 2: answer_question - Researching answers")
        q = ev.question
        if not q.strip():
            ctx.write_event_to_stream(ProgressEvent(progress="Skipping empty question", step="Question Skipped"))
            logger.info(f"Skipping empty question")
            return None
        ctx.write_event_to_stream(ProgressEvent(
            progress=f"Researching answer...", 
            step="Researching"
        ))
        logger.info(f"Researching answer for: {q[:50]}...")
        agent = FunctionAgent(tools=await ctx.get("tools"), llm=Settings.llm)
        resp = await agent.run(q)
        logger.info("Answer found")
        ctx.write_event_to_stream(ProgressEvent(
            progress=f"Answer found", 
            step="Research Complete"
        ))
        logger.info(f"📚 STEP 2 COMPLETE: answer_question - Answer found")
        return AnswerEvent(question=q, answer=str(resp))

    @step
    async def write_report(self, ctx: Context, ev: AnswerEvent) -> ReviewEvent:
        logger.info(f"✍️ STEP 3: write_report - Drafting content")
        total = await ctx.get("num_questions")
        answers = ctx.collect_events(ev, [AnswerEvent] * total)
        if not answers:
            logger.warning("No answers collected, cannot generate study guide")
            logger.info(f"write_report step ended (no answers)")
            return None
        prev = await ctx.get("previous", [])
        prev.extend(answers)
        await ctx.set("previous", prev)
        outline = await ctx.get("outline")
        ctx.write_event_to_stream(ProgressEvent(
            progress="Drafting study guide...", 
            step="Writing Draft"
        ))
        logger.info(f"Writing study guide draft with {len(answers)} answers")
        prompt = f"Compose a comprehensive study guide based on this outline: {outline}\n"
        for ans in prev:
            prompt += f"Q: {ans.question}\nA: {ans.answer}\n"
        resp = await Settings.llm.acomplete(prompt)
        ctx.write_event_to_stream(ProgressEvent(
            progress="Initial study guide draft completed", 
            step="Draft Complete"
        ))
        logger.info("Study guide draft completed")
        logger.info(f"✍️ STEP 3 COMPLETE: write_report - Draft completed")
        return ReviewEvent(study_guide=str(resp))

    @step
    async def review_report(self, ctx: Context, ev: ReviewEvent) -> Union[StopEvent, QuestionEvent]:
        logger.info(f"🔄 STEP 4: review_report - Reviewing and refining content")
        count = await ctx.get("num_reviews", 0) + 1
        await ctx.set("num_reviews", count)
        ctx.write_event_to_stream(ProgressEvent(
            progress=f"Reviewing draft (round {count})...", 
            step="Reviewing"
        ))
        logger.info(f"Reviewing study guide draft - round {count}")
        prompt = (
            f"Review the study guide for topic '{await ctx.get('original_query')}'. "
            "If it adequately covers the topic with sufficient detail for learning, return 'OKAY'. Otherwise, list up to 2 concise questions to improve it."
        )
        resp = await Settings.llm.acomplete(prompt)
        text = str(resp)
        if text.strip() == "OKAY" or count >= 3:
            ctx.write_event_to_stream(ProgressEvent(
                progress="Study guide approved!", 
                step="Completed"
            ))
            logger.info("Study guide approved")
            logger.info(f"🔄 STEP 4 COMPLETE: review_report - Content approved")
            return StopEvent(result=ev.study_guide)
        logger.info(f"Study guide needs improvement - generating follow-up questions (round {count})")
        questions = [q for q in text.splitlines() if q][:2]
        await ctx.set("num_questions", len(questions))
        for i, q in enumerate(questions):
            logger.info(f"Follow-up question {i+1}: {q[:50]}...")
            ctx.write_event_to_stream(ProgressEvent(
                progress=f"Additional question {i+1}/{len(questions)} for improvement", 
                step="Follow-up"
            ))
            ctx.send_event(QuestionEvent(question=q))
        logger.info(f"🔄 STEP 4 ITERATING: review_report - Generated {len(questions)} follow-up questions")

async def process_document(file_path, query, storage_dir):
    """Process the document and generate a study guide"""
    # Reset progress
    st.session_state.progress_updates = []
    st.session_state.study_guide = ""
    
    # Add a log handler that writes to the UI
    log_container = st.empty()
    logs = []
    
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            logs.append(log_entry)
            # Only show the last 10 logs to keep it concise
            log_container.code("\n".join(logs[-10:]), language="bash")
    
    # Add the custom handler
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
    
    try:
        # Log start of processing
        logger.info(f"🚀 Starting document processing for query: {query}")
        logger.info(f"📄 Document: {os.path.basename(file_path)}")
        
        # Configure the models
        configure_models()
        
        # Get index and create tool
        with st.status("Creating document index..."):
            try:
                index = get_index(file_path, storage_dir)
            except Exception as e:
                error_msg = f"Error creating document index: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.progress_updates.append({"progress": error_msg, "step": "Error"})
                st.session_state.processing = False
                return None
        
        logger.info("Creating research tool from document index")
        doc_tool = make_document_tool(
            index, 
            name="document_analysis",
            description=f"Detailed information from the uploaded document"
        )
        
        # Skip the workflow and use the direct approach
        logger.info("Using direct content generation approach")
        st.session_state.progress_updates.append({"progress": "Starting content generation", "step": "Initializing"})
        
        try:
            # Generate content directly 
            content = await generate_content_direct(query, doc_tool)
            logger.info("✅ Content generation completed successfully!")
            st.session_state.study_guide = content
            st.session_state.progress_updates.append({"progress": "Study guide generated successfully", "step": "Completed"})
            return content
                
        except Exception as e:
            error_msg = f"Content generation error: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.session_state.progress_updates.append({"progress": error_msg, "step": "Error"})
            raise e
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.exception("Error during document processing")
        st.error(error_msg)
        st.session_state.progress_updates.append({"progress": error_msg, "step": "Error"})
        return None
    finally:
        # Remove the handler
        logger.removeHandler(handler)
        st.session_state.processing = False

# Direct content generation function
async def generate_content_direct(query, doc_tool):
    """Generate study guide content directly without using the workflow"""
    engine = doc_tool.query_engine
    
    # STEP 1: Formulate Questions (Create an outline with questions)
    logger.info("🔍 STEP 1: Formulate Questions - Creating outline and research questions")
    st.session_state.progress_updates.append({"progress": "Generating research questions...", "step": "Creating Questions"})
    
    # Log when RAG is being used - Vector store query happening here
    logger.info("RAG QUERY: Searching vector store for outline information...")
    st.session_state.progress_updates.append({"progress": "Searching document for relevant outline information...", "step": "Vector Search"})
    
    outline_query = f"Create a detailed outline with 3-5 key questions for a study guide about: {query}"
    outline_response = await engine.aquery(outline_query)
    outline = outline_response.response
    logger.info(f"RAG complete: Retrieved {len(outline_response.source_nodes)} chunks from vector store")
    
    # Extract questions from outline
    import re
    questions = re.findall(r'\d+\.\s*(.*?\?)', outline, re.DOTALL)
    if not questions:
        # If no questions found, generate them explicitly
        question_query = f"Based on this outline: {outline}\n\nGenerate 3-5 key questions for a study guide about: {query}"
        question_response = await engine.aquery(question_query)
        questions = re.findall(r'\d+\.\s*(.*?\?)', question_response.response, re.DOTALL)
        if not questions:
            questions = question_response.response.splitlines()
            
    # Log the questions
    for i, q in enumerate(questions[:5]):
        logger.info(f"Question {i+1}: {q[:100]}...")
        st.session_state.progress_updates.append({"progress": f"Generated question {i+1}", "step": "Question Created"})
    
    logger.info(f"🔍 STEP 1 COMPLETE: Formulate Questions - Generated outline and {len(questions)} questions")
    
    # STEP 2: Answer Questions
    logger.info("📚 STEP 2: Answer Questions - Researching answers from document")
    st.session_state.progress_updates.append({"progress": "Researching answers to questions...", "step": "Researching"})
    
    # Log when RAG is being used - Vector store query happening here
    logger.info("RAG QUERY: Searching vector store for detailed answers...")
    st.session_state.progress_updates.append({"progress": "Searching document for answers...", "step": "Vector Search"})
    
    answers = []
    for i, question in enumerate(questions[:5]):
        logger.info(f"Researching answer for question {i+1}: {question[:100]}...")
        answer_response = await engine.aquery(f"Answer this question in detail: {question}")
        answer = answer_response.response
        answers.append({"question": question, "answer": answer})
        st.session_state.progress_updates.append({"progress": f"Found answer for question {i+1}", "step": "Research Complete"})
    
    logger.info(f"📚 STEP 2 COMPLETE: Answer Questions - Found answers for {len(answers)} questions")
    
    # STEP 3: Write Report
    logger.info("✍️ STEP 3: Write Study Guide - Drafting study guide content")
    st.session_state.progress_updates.append({"progress": "Drafting study guide content...", "step": "Writing Draft"})
    
    # Create content from questions and answers
    content_parts = [f"## Outline\n{outline}"]
    
    for i, qa in enumerate(answers):
        content_parts.append(f"\n## Question {i+1}: {qa['question']}\n\n{qa['answer']}")
    
    content = "\n".join(content_parts)
    logger.info("Study guide draft completed")
    st.session_state.progress_updates.append({"progress": "Initial study guide draft completed", "step": "Draft Complete"})
    logger.info(f"✍️ STEP 3 COMPLETE: Write Study Guide - Study guide draft completed")
    
    # STEP 4: Review & Refine
    logger.info("🔄 STEP 4: Review & Refine - Finalizing study guide with summary and improvements")
    st.session_state.progress_updates.append({"progress": "Reviewing and refining study guide...", "step": "Reviewing"})
    
    # Log when RAG is being used - Vector store query happening here
    logger.info("RAG QUERY: Searching vector store for summary and review information...")
    st.session_state.progress_updates.append({"progress": "Searching document for summary information...", "step": "Vector Search"})
    
    final_query = f"""
    Review and improve this study guide:
    {content}
    
    Add:
    1. A brief introduction at the beginning
    2. A concise summary at the end
    3. 3-5 review questions for self-assessment
    4. Key terms and definitions section
    5. Study tips related to this topic
    
    Format the final study guide clearly with markdown.
    """
    
    final_response = await engine.aquery(final_query)
    final_content = final_response.response
    logger.info(f"RAG complete: Retrieved {len(final_response.source_nodes)} chunks from vector store")
    logger.info("Study guide review and refinement completed")
    st.session_state.progress_updates.append({"progress": "Study guide finalized!", "step": "Completed"})
    
    logger.info(f"🔄 STEP 4 COMPLETE: Review & Refine - Study guide finalized")
    
    # Add source references
    source_references = "\n\n## Sources Used\n"
    all_sources = set()
    
    # Collect unique sources from all responses
    for node in outline_response.source_nodes:
        if hasattr(node, 'metadata') and 'file_name' in node.metadata:
            source = f"- {node.metadata.get('file_name', 'Unknown')}"
            all_sources.add(source)
        elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
            source = f"- {node.node.metadata.get('file_name', 'Unknown')}"
            all_sources.add(source)
    
    # Add sources to content if any were found
    if all_sources:
        source_references += "\n".join(all_sources)
        logger.info(f"Added {len(all_sources)} source references to the content")
    else:
        source_references = ""
    
    # Format the final content
    formatted_content = f"# Study Guide: {query}\n\n{final_content}{source_references}"
    
    # Display RAG statistics
    total_chunks = len(outline_response.source_nodes) + len(final_response.source_nodes)
    for answer in answers:
        # Need to check if response has source_nodes attribute
        if hasattr(answer.get("response", None), "source_nodes"):
            total_chunks += len(answer.get("response").source_nodes)
    
    logger.info(f"📊 RAG STATS: Total chunks retrieved from vector store: {total_chunks}")
    st.session_state.progress_updates.append({"progress": f"Retrieved chunks from your document", "step": "Statistics"})
    
    return formatted_content

def handle_file_upload():
    """Handle file upload and save to temporary location"""
    if st.session_state.uploaded_file:
        # Create a temporary directory for the uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        # Create a storage directory for the index
        storage_dir = os.path.join(temp_dir, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Store the paths in session state
        st.session_state.uploaded_file_path = file_path
        st.session_state.storage_dir = storage_dir
        return True
    return False

def main():
    st.title("📚 Study Guide Generator")
    
    # Show a status indicator if processing
    if st.session_state.processing:
        st.warning("⏳ Processing your request... Please wait.", icon="⏳")
    
    st.markdown("""
    Upload your academic document (PDF) and enter a topic to generate a comprehensive study guide.
    The assistant will analyze your document and create a structured study guide based on your topic.
    """)
    
    # Add explanation of main steps
    with st.expander("How the process works"):
        st.markdown("""
        ### Main Steps of the Study Guide Generation Process:
        
        1. 🔍 **Formulate Questions**: Generate key questions about the topic
        2. 📚 **Answer Questions**: Research answers from your academic document 
        3. ✍️ **Write Guide**: Draft study materials based on research
        4. 🔄 **Review & Refine**: Review the content and make improvements
        """)
    
    # File uploader with better UI
    with st.container():
        st.subheader("1. Upload Academic Document")
        uploaded_file = st.file_uploader(
            "Upload a PDF document", 
            type=["pdf"], 
            key="uploaded_file", 
            on_change=handle_file_upload,
            help="Upload a PDF document to analyze",
            disabled=st.session_state.processing
        )
    
    # Only show the rest if a file is uploaded
    if st.session_state.uploaded_file_path:
        st.success(f"✅ File uploaded: {os.path.basename(st.session_state.uploaded_file_path)}")
        
        # Query input
        st.subheader("2. Enter Study Topic")
        query = st.text_input(
            "What topic would you like to create a study guide for?",
            placeholder="E.g., 'Mitochondrial function', 'Quantum mechanics fundamentals'",
            help="Enter a topic to create a study guide from your academic document",
            disabled=st.session_state.processing
        )
        
        # Process button
        if st.button(
            "🚀 Generate Study Guide", 
            disabled=st.session_state.processing,
            type="primary",
            use_container_width=True
        ):
            if query:
                st.session_state.processing = True
                
                # Initialize RAG stats in session state
                if "rag_stats" not in st.session_state:
                    st.session_state.rag_stats = {
                        "chunks_retrieved": 0,
                        "vector_store_size": 0,
                        "queries_made": 0,
                        "rag_operations": []
                    }
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Progress", "System Logs", "RAG Info"])
                
                with tab1:
                    # Create placeholder for progress
                    st.info("Processing document and generating study guide...")
                    progress_placeholder = st.empty()
                
                with tab2:
                    # Placeholder for logs
                    st.info("System logs will appear here")
                    log_placeholder = st.empty()
                
                with tab3:
                    # Placeholder for RAG info
                    st.info("RAG (Retrieval-Augmented Generation) process details will appear here")
                    rag_placeholder = st.empty()
                    rag_placeholder.markdown("""
                    ### RAG Process
                    1. **Document Chunking**: Document is split into chunks
                    2. **Embedding Creation**: Each chunk is converted to a vector embedding
                    3. **Vector Storage**: Embeddings are stored in a vector database
                    4. **Semantic Retrieval**: When queried, relevant chunks are retrieved based on semantic similarity
                    5. **Content Generation**: Retrieved information is used to generate content
                    """)
                
                # Run the process
                loop = asyncio.get_event_loop()
                loop.run_until_complete(process_document(
                    st.session_state.uploaded_file_path,
                    query,
                    st.session_state.storage_dir
                ))
                
                # Force a rerun after processing is complete to update UI
                st.rerun()
            else:
                st.error("Please enter a study topic")
        
        # Display progress
        if st.session_state.progress_updates:
            tab1, tab2, tab3 = st.tabs(["Progress", "System Logs", "RAG Info"])
            
            with tab1:
                st.subheader("3. Workflow Progress")
                
                # Show main workflow steps
                st.markdown("""
                ### Main Workflow Steps:
                
                1. 🔍 **Formulate Questions** - Generate key research questions
                2. 📚 **Answer Questions** - Research document for answers
                3. ✍️ **Write Guide** - Create content from research
                4. 🔄 **Review & Refine** - Review and improve content
                """)
                
                # Show visual progress tracker
                progress_steps = ["Outline", "Vector Search", "Content", "Summary", "Completed"]
                completed_steps = []
                for update in st.session_state.progress_updates:
                    if isinstance(update, dict) and update.get("step") in progress_steps and update.get("step") not in completed_steps:
                        completed_steps.append(update.get("step"))
                
                # Display progress bar
                if completed_steps:
                    progress_value = len(completed_steps) / len(progress_steps)
                    st.progress(progress_value)
                    
                    # Show current step or completion
                    if len(completed_steps) < len(progress_steps):
                        current_step_index = len(completed_steps)
                        if current_step_index < len(progress_steps):
                            current_step = progress_steps[current_step_index]
                            st.info(f"Current step: {current_step}")
                    else:
                        st.success("✅ All steps completed")
                
                # Display detailed progress updates in an expander
                with st.expander("View detailed progress", expanded=True):
                    # Filter and highlight main steps in logs
                    main_steps = []
                    other_updates = []
                    
                    for update in st.session_state.progress_updates:
                        if isinstance(update, dict):
                            step = update.get("step", "")
                            progress = update.get("progress", "")
                            
                            # Check if this is related to a main step
                            if "Question Created" in step or "Creating Questions" in step:
                                main_steps.append(("🔍 STEP 1: Formulate Questions", progress))
                            elif "Research" in step:
                                main_steps.append(("📚 STEP 2: Answer Questions", progress))
                            elif "Writing Draft" in step or "Draft Complete" in step:
                                main_steps.append(("✍️ STEP 3: Write Guide", progress))
                            elif "Reviewing" in step or "Follow-up" in step or "Completed" in step:
                                main_steps.append(("🔄 STEP 4: Review & Refine", progress))
                            else:
                                other_updates.append((step, progress))
                        else:
                            other_updates.append(("Info", str(update)))
                    
                    # Display main steps with emphasis
                    if main_steps:
                        st.markdown("#### Main Workflow Steps:")
                        for step, progress in main_steps:
                            st.success(f"**{step}**: {progress}")
                        
                    # Display other updates
                    if other_updates:
                        st.markdown("#### Other Progress Updates:")
                        for step, progress in other_updates:
                            if step == "Error":
                                st.error(f"{progress}")
                            elif "Vector Search" in step:
                                st.info(f"🔍 {progress}")
                            else:
                                st.info(f"{step}: {progress}")
            
            with tab2:
                st.subheader("System Logs")
                st.info("These logs show the technical details of the process")
                
                # Create a container for the logs
                with st.container():
                    # Filter out log entries that contain sensitive data or are too verbose
                    log_entries = []
                    for update in st.session_state.progress_updates:
                        if isinstance(update, dict):
                            log_entries.append(f"{update.get('step', 'Info')}: {update.get('progress', '')}")
                    
                    # Display the logs in a code block
                    if log_entries:
                        st.code("\n".join(log_entries), language="bash")
            
            with tab3:
                st.subheader("RAG Process Information")
                
                # Extract RAG-related updates
                rag_updates = [update for update in st.session_state.progress_updates 
                               if isinstance(update, dict) and 
                               ("Vector Search" in update.get("step", "") or 
                                "Statistics" in update.get("step", ""))]
                
                # Count occurrences of vector searches
                vector_search_count = sum(1 for update in rag_updates if "Vector Search" in update.get("step", ""))
                
                # Find statistics update if available
                stats_update = next((update for update in rag_updates if "Statistics" in update.get("step", "")), None)
                chunks_retrieved = 0
                if stats_update:
                    progress_text = stats_update.get("progress", "")
                    # Try to extract the number of chunks
                    import re
                    match = re.search(r"Retrieved (\d+) relevant chunks", progress_text)
                    if match:
                        chunks_retrieved = int(match.group(1))
                
                # Display RAG summary
                st.markdown(f"""
                ### RAG Summary
                - **Vector Searches Performed**: {vector_search_count}
                - **Document Chunks Retrieved**: {chunks_retrieved}
                """)
                
                # Display information about vector store reuse if applicable
                if "rag_stats" in st.session_state:
                    if st.session_state.rag_stats.get("reused_vector_store", False):
                        st.success(f"""
                        ### ⚡ Using Existing Vector Store
                        This run is using a pre-computed vector store from previous runs, which saves time 
                        and computational resources. No need to reprocess the document!
                        """)
                    elif st.session_state.rag_stats.get("new_vector_store", False):
                        chunks_count = st.session_state.rag_stats.get("chunks_count", 0)
                        st.info(f"""
                        ### 🆕 Created New Vector Store
                        A new vector store was created for this document with {chunks_count} chunks.
                        This vector store will be reused in future runs with the same document.
                        """)
                
                # Display RAG operations
                st.markdown("### RAG Operations")
                for update in rag_updates:
                    step = update.get("step", "")
                    progress = update.get("progress", "")
                    
                    if "Vector Search" in step:
                        st.info(f"🔍 {progress}")
                    elif "Statistics" in step:
                        st.success(f"📊 {progress}")
                
                # Show visual explanation of RAG
                st.markdown("### How RAG Works")
                st.markdown("""
                1. **Document Processing**:
                   - PDF document is parsed and split into chunks
                   - Each chunk is embedded (converted to vectors)
                   - Vectors are stored in a searchable database
                
                2. **Query Processing**:
                   - When you ask a question, it's converted to an embedding
                   - Similar chunks from your document are retrieved
                   - The LLM generates content using both your query and the retrieved information
                """)
        
        # Display result
        if st.session_state.study_guide:
            st.subheader("4. Generated Study Guide")
            st.markdown("---")
            
            # Create a styled container for the study guide
            with st.container():
                # Use st.write() instead of unsafe_allow_html for better reliability
                st.markdown("**Generated Study Guide:**")
                st.write(st.session_state.study_guide)
            
            st.markdown("---")
            
            # Allow downloading the study guide
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download as Markdown",
                    data=st.session_state.study_guide,
                    file_name="generated_study_guide.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    label="📄 Download as Text",
                    data=st.session_state.study_guide,
                    file_name="generated_study_guide.txt",
                    mime="text/plain"
                )
    
    # Show instructions if no file uploaded
    else:
        st.info("👆 Please upload a PDF document to get started")

if __name__ == "__main__":
    main() 