#streamlit_app.py
#streamlit run streamlit_app1.py

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import sys
import argparse
import torch
import gc


_original_read_csv = pd.read_csv
_original_read_excel = pd.read_excel

def _patched_read_csv(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_read_csv(*args, **kwargs)

def _patched_read_excel(*args, **kwargs):
    return _original_read_excel(*args, **kwargs)

pd.read_csv = _patched_read_csv
pd.read_excel = _patched_read_excel

# Patch built-in open for JSON files
import builtins
_original_open = builtins.open

def _patched_open(file, mode='r', *args, **kwargs):
    if 'r' in mode and 'b' not in mode:  # Text read mode
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _patched_open


# Patch json.load
_original_json_load = json.load

def _patched_json_load(fp, *args, **kwargs):
    # Ensure file is opened with UTF-8 if it's a file path
    if isinstance(fp, str):
        with open(fp, 'r', encoding='utf-8') as f:
            return _original_json_load(f, *args, **kwargs)
    return _original_json_load(fp, *args, **kwargs)
# ⭐⭐⭐ END OF PATCH ⭐⭐⭐



# Add the parent directory to path to import the main model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conversational_rag.inference_rag import (
    init_qwen_model,
    ConversationRAG,
    generate_response,
    create_qwen_rag_prompt
)

# Page configuration
st.set_page_config(
    page_title="AI Assistant Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for RAG chatbot
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Initialize session state for Schema-Action chatbot
if 'schema_system' not in st.session_state:
    st.session_state.schema_system = None

if 'schema_chat_history' not in st.session_state:
    st.session_state.schema_chat_history = []

if 'schema_model_loaded' not in st.session_state:
    st.session_state.schema_model_loaded = False

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# File paths
SATISFIED_FILE = "./feedback/satisfied_responses.xlsx"
UNSATISFIED_FILE = "./feedback/unsatisfied_responses.xlsx"
RAG_DATASET = "./conversational_rag/examples/lv_customer_service/data/knowledge.jsonl"

# Create directories if they don't exist
os.makedirs("./feedback", exist_ok=True)


def save_feedback_to_excel(question, response, satisfied, correct_answer=None):
    """Save feedback to Excel file"""
    feedback_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'response': response,
        'satisfied': satisfied,
        'correct_answer': correct_answer if correct_answer else ""
    }
    
    file_path = SATISFIED_FILE if satisfied else UNSATISFIED_FILE
    
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = pd.concat([df, pd.DataFrame([feedback_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_entry])
    
    df.to_excel(file_path, index=False)
    return True


def add_to_rag_dataset(question, answer):
    """Add a conversation pair to the RAG dataset"""
    conversation_entry = {
        "conversations": [
            {"content": question, "role": "user"},
            {"content": answer, "role": "assistant"}
        ]
    }
    
    with open(RAG_DATASET, 'a', encoding='utf-8') as f:
        f.write(json.dumps(conversation_entry, ensure_ascii=False) + '\n')
    
    return True


@st.cache_resource
def load_model_and_rag(lora_path, rag_dataset, rag_threshold):
    """Load model and RAG system - cached to avoid reloading"""
    args = argparse.Namespace(
        lora_path=lora_path,
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=150,
        max_seq_len=1024,
        rag_threshold=rag_threshold,
        enable_thinking=True,
        seed=42
    )
    
    model, tokenizer = init_qwen_model(args)
    
    rag_system = None
    if os.path.exists(rag_dataset):
        rag_system = ConversationRAG()
        if rag_system.build_index(rag_dataset):
            st.success("✅ RAG system loaded successfully!")
        else:
            st.warning("⚠️ Failed to load RAG system")
    else:
        st.warning(f"⚠️ RAG dataset not found: {rag_dataset}")
    
    return model, tokenizer, rag_system, args


def reinitialize_rag(rag_dataset, rag_threshold):
    """Reinitialize RAG system after adding new data"""
    if os.path.exists(rag_dataset):
        rag_system = ConversationRAG()
        if rag_system.build_index(rag_dataset):
            st.session_state.rag_system = rag_system
            st.success("✅ RAG system reinitialized successfully!")
            return True
    return False


def check_rag_reinit_signal():
    """Check if admin panel has requested RAG reinitialization"""
    signal_file = "./feedback/rag_reinit_signal.txt"
    if os.path.exists(signal_file):
        try:
            with open(signal_file, 'r', encoding='utf-8') as f:
                signal_time = f.read().strip()
            
            if 'last_rag_reinit' not in st.session_state or st.session_state.last_rag_reinit != signal_time:
                st.session_state.last_rag_reinit = signal_time
                os.remove(signal_file)
                return True
        except:
            pass
    return False


def unmerge_lora_and_clear():
    """Unmerge LoRA adapters and clear model from memory"""
    if st.session_state.model is not None:
        try:
            from peft import PeftModel
            
            if isinstance(st.session_state.model, PeftModel):
                st.session_state.model = st.session_state.model.unmerge_adapter()
            
            # Clear all model-related session state
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.rag_system = None
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            st.error(f"Error clearing model: {e}")
            return False
    return False


@st.cache_resource
def load_schema_system(data_directory, metadata_path, join_config_path, use_4bit):
    """Load the Schema-Action RAG system"""
    try:
        # ⭐ APPLY PATCH INSIDE THE FUNCTION BEFORE IMPORT ⭐
        import pandas as pd
        import builtins
        
        # Patch pandas.read_csv
        _original_read_csv = pd.read_csv
        def _patched_read_csv(*args, **kwargs):
            if 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return _original_read_csv(*args, **kwargs)
        pd.read_csv = _patched_read_csv
        
        # Patch built-in open
        _original_open = builtins.open
        def _patched_open(file, mode='r', *args, **kwargs):
            if 'r' in mode and 'b' not in mode and 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return _original_open(file, mode, *args, **kwargs)
        builtins.open = _patched_open
        # ⭐ END PATCH ⭐
        
        from schema_action.schema_action import SchemaActionRAGSystem
        
        args = argparse.Namespace(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_4bit=True,
            cache_size=1000,
            debug=True,
            data_directory=data_directory,
            metadata_path=metadata_path,
            join_config_path=join_config_path,
            seed=42
        )
        
        with st.spinner("Loading Schema-Action system..."):
            system = SchemaActionRAGSystem(args)
        
        return system, None
    except UnicodeDecodeError as e:
        return None, f"Encoding error: {e}\n\nPlease ensure all files are UTF-8 encoded."
    except Exception as e:
        return None, str(e)

def rag_chatbot_tab():
    """RAG Chatbot Tab Content"""
    st.header("🎒 Louis Vuitton OnTheGo Customer Service")
    
    # Check for RAG reinitialization signal
    if check_rag_reinit_signal() and st.session_state.model is not None:
        with st.spinner("Reloading RAG system with new data..."):
            if reinitialize_rag(
                st.session_state.get('rag_dataset_path', RAG_DATASET),
                st.session_state.args.rag_threshold
            ):
                st.toast("🔄 RAG system updated!", icon="✅")
    
    # Model controls
    with st.expander("⚙️ Model Configuration", expanded=not st.session_state.model):
        col1, col2 = st.columns(2)
        
        with col1:
            lora_path = st.text_input(
                "LoRA Path", 
                value="conversational_rag/examples/lv_customer_service/models/lv_lora"
            )
            
            temperature = st.slider(
                "Temperature", 
                0.0, 2.0, 0.7, 0.1
            )
            
            top_p = st.slider(
                "Top P", 
                0.0, 1.0, 0.8, 0.05
            )
        
        with col2:
            max_new_tokens = st.slider(
                "Max New Tokens", 
                50, 500, 150, 10
            )
            
            use_rag = st.checkbox("Enable RAG", value=True)
            
            rag_threshold = st.slider(
                "RAG Threshold", 
                0.0, 1.0, 0.3, 0.05
            )
        
        rag_dataset = st.text_input(
            "RAG Dataset Path",
            value="./conversational_rag/examples/lv_customer_service/data/knowledge.jsonl"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load/Reload Model", type="primary", use_container_width=True):
                with st.spinner("Loading model and RAG system..."):
                    model, tokenizer, rag_system, args = load_model_and_rag(
                        lora_path, rag_dataset, rag_threshold
                    )
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.rag_system = rag_system
                    st.session_state.args = args
                    st.session_state.rag_dataset_path = rag_dataset
                    st.success("✅ Model loaded!")
        
        with col2:
            if st.button("🗑️ Unload Model", use_container_width=True):
                if unmerge_lora_and_clear():
                    st.success("✅ Model unloaded!")
                    st.rerun()
    
    # Model status
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model first!")
        return
    
    # Update args with current parameters
    if hasattr(st.session_state, 'args'):
        st.session_state.args.temperature = temperature
        st.session_state.args.top_p = top_p
        st.session_state.args.max_new_tokens = max_new_tokens
        st.session_state.args.rag_threshold = rag_threshold
    
    # Chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['response'])
            
            if chat.get('rag_context'):
                with st.expander("📚 Retrieved Context"):
                    for j, ctx in enumerate(chat['rag_context'], 1):
                        st.write(f"**{j}. Score: {ctx['score']:.3f}**")
                        st.write(f"Q: {ctx['question']}")
                        st.write(f"A: {ctx['answer'][:200]}...")
            
            if not chat.get('feedback_given', False):
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"✅ Satisfied", key=f"sat_{i}"):
                        save_feedback_to_excel(chat['question'], chat['response'], True)
                        st.session_state.chat_history[i]['feedback_given'] = True
                        st.success("Thanks for feedback!")
                        st.rerun()
                
                with col2:
                    if st.button(f"❌ Not Satisfied", key=f"unsat_{i}"):
                        st.session_state.current_feedback_idx = i
                        st.session_state.show_correction_form = True
                        st.rerun()
                
                with col3:
                    if st.button(f"➕ Add to Knowledge", key=f"add_{i}"):
                        st.session_state.current_knowledge_idx = i
                        st.session_state.show_knowledge_form = True
                        st.rerun()
            
            # Correction form
            if (st.session_state.get('show_correction_form') and 
                st.session_state.get('current_feedback_idx') == i):
                st.write("---")
                correct_answer = st.text_area("Correct Answer:", height=150, key=f"corr_{i}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit", key=f"submit_corr_{i}"):
                        if correct_answer.strip():
                            save_feedback_to_excel(chat['question'], chat['response'], False, correct_answer)
                            st.session_state.chat_history[i]['feedback_given'] = True
                            st.session_state.show_correction_form = False
                            st.success("Thanks!")
                            st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_corr_{i}"):
                        st.session_state.show_correction_form = False
                        st.rerun()
            
            # Knowledge form
            if (st.session_state.get('show_knowledge_form') and 
                st.session_state.get('current_knowledge_idx') == i):
                st.write("---")
                st.info("💡 Add to knowledge base")
                knowledge_q = st.text_area("Question:", value=chat['question'], height=100, key=f"kq_{i}")
                knowledge_a = st.text_area("Answer:", value=chat['response'], height=150, key=f"ka_{i}")
                update_rag = st.checkbox("🔄 Update RAG immediately", value=True, key=f"ur_{i}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("➕ Add", key=f"add_kb_{i}", type="primary"):
                        if knowledge_q.strip() and knowledge_a.strip():
                            add_to_rag_dataset(knowledge_q, knowledge_a)
                            st.session_state.chat_history[i]['feedback_given'] = True
                            st.session_state.show_knowledge_form = False
                            
                            if update_rag and st.session_state.rag_system:
                                with st.spinner("Updating RAG..."):
                                    if reinitialize_rag(RAG_DATASET, rag_threshold):
                                        st.success("✅ Added & RAG updated!")
                                    else:
                                        st.warning("Added but RAG reload failed")
                            else:
                                st.success("✅ Added to knowledge base!")
                            st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_kb_{i}"):
                        st.session_state.show_knowledge_form = False
                        st.rerun()
    
    # Chat input
    user_input = st.chat_input("Ask about Louis Vuitton OnTheGo bags...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    user_input,
                    st.session_state.rag_system if use_rag else None,
                    st.session_state.args,
                    use_rag=use_rag
                )
                st.write(response)
                
                rag_context = []
                if use_rag and st.session_state.rag_system:
                    retrieved = st.session_state.rag_system.retrieve(user_input, top_k=3, threshold=rag_threshold)
                    if retrieved:
                        rag_context = [
                            {'score': float(score), 'question': conv_pair.question, 'answer': conv_pair.answer}
                            for conv_pair, score in retrieved
                        ]
                        with st.expander("📚 Retrieved Context"):
                            for j, ctx in enumerate(rag_context, 1):
                                st.write(f"**{j}. Score: {ctx['score']:.3f}**")
                                st.write(f"Q: {ctx['question']}")
                                st.write(f"A: {ctx['answer'][:200]}...")
        
        st.session_state.chat_history.append({
            'question': user_input,
            'response': response,
            'rag_context': rag_context,
            'feedback_given': False,
            'timestamp': datetime.now()
        })
        st.rerun()


def schema_action_tab():
    """Schema-Action Chatbot Tab Content"""
    st.header("📊 Schema-Action Query System")
    
    # Configuration
    with st.expander("⚙️ System Configuration", expanded=not st.session_state.schema_model_loaded):
        col1, col2 = st.columns(2)
        
        with col1:
            data_directory = st.text_input("Data Directory", value="./schema_action/examples/retail_analytics/data/")
            metadata_path = st.text_input("Metadata Path", value="./schema_action/configs/metadata.json")
        
        with col2:
            join_config_path = st.text_input("Join Config Path", value="./schema_action/configs/join_config.json")
            use_4bit = st.checkbox("Use 4-bit Quantization", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Unmerge LoRA & Clear", use_container_width=True):
                with st.spinner("Clearing..."):
                    if unmerge_lora_and_clear():
                        st.success("✅ Cleared!")
                    else:
                        st.info("Nothing to clear")
        
        with col2:
            if st.button("🚀 Load Schema System", type="primary", use_container_width=True):
                if not os.path.exists(data_directory):
                    st.error(f"❌ Directory not found: {data_directory}")
                elif not os.path.exists(metadata_path):
                    st.error(f"❌ Metadata not found: {metadata_path}")
                elif not os.path.exists(join_config_path):
                    st.error(f"❌ Join config not found: {join_config_path}")
                else:
                    unmerge_lora_and_clear()
                    system, error = load_schema_system(data_directory, metadata_path, join_config_path, use_4bit)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.session_state.schema_system = system
                        st.session_state.schema_model_loaded = True
                        st.session_state.schema_data_dir = data_directory
                        st.success("✅ System loaded!")
    
    # System status
    if not st.session_state.schema_model_loaded:
        st.info("👆 Please load the Schema-Action system first!")
        
        with st.expander("📖 Setup Guide"):
            st.markdown("""
            ### Required Files:
            1. **Data Directory**: CSV/Excel files
            2. **metadata.json**: File descriptions
            3. **join_config.json**: Table relationships
            
            See SCHEMA_ACTION_GUIDE.md for details.
            """)
        return
    
    # Chat history
    for i, chat in enumerate(st.session_state.schema_chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {chat['processing_time']}s")
            with col2:
                st.caption(f"🎯 {chat['confidence']:.2%}")
            with col3:
                if 'selected_tables' in chat:
                    st.caption(f"📁 {len(chat['selected_tables'])} tables")
            
            if chat.get('selected_tables'):
                with st.expander("📊 Tables Used"):
                    for table in chat['selected_tables']:
                        st.write(f"• {table}")
            
            if chat.get('data_result_count', 0) > 0:
                st.info(f"Found {chat['data_result_count']} records")
    
    # Chat input
    user_query = st.chat_input("Ask a question about your data...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = st.session_state.schema_system.process_query(
                    user_query,
                    st.session_state.schema_data_dir
                )
            
            st.write(result['answer'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {result['processing_time']}s")
            with col2:
                st.caption(f"🎯 {result['confidence']:.2%}")
            with col3:
                if 'selected_tables' in result:
                    st.caption(f"📁 {len(result['selected_tables'])} tables")
            
            if result.get('selected_tables'):
                with st.expander("📊 Tables Used"):
                    for table in result['selected_tables']:
                        st.write(f"• {table}")
            
            if result.get('data_result_count', 0) > 0:
                st.info(f"Found {result['data_result_count']} records")
        
        st.session_state.schema_chat_history.append({
            'question': user_query,
            'answer': result['answer'],
            'processing_time': result['processing_time'],
            'confidence': result['confidence'],
            'selected_tables': result.get('selected_tables', []),
            'data_result_count': result.get('data_result_count', 0),
            'timestamp': datetime.now()
        })
        st.rerun()


def main():
    st.title("🤖 Multi-Purpose AI Assistant")
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 System Status")
        
        # RAG Chatbot status
        st.subheader("🎒 LV Chatbot")
        if st.session_state.model:
            st.success("✅ Active")
            st.metric("Conversations", len(st.session_state.chat_history))
        else:
            st.info("⚪ Not Loaded")
        
        # Schema-Action status
        st.subheader("📊 Schema Query")
        if st.session_state.schema_model_loaded:
            st.success("✅ Active")
            st.metric("Queries", len(st.session_state.schema_chat_history))
            if st.session_state.schema_system:
                cache_stats = st.session_state.schema_system.cache.stats()
                st.metric("Cache Hit Rate", cache_stats['hit_rate'])
        else:
            st.info("⚪ Not Loaded")
        
        st.divider()
        
        
        # Memory management
        if st.session_state.model or st.session_state.schema_system:
            st.divider()
            st.subheader("🧹 Memory Management")
            if st.button("🗑️ Clear All Models", use_container_width=True, type="secondary"):
                unmerge_lora_and_clear()
                st.session_state.schema_system = None
                st.session_state.schema_model_loaded = False
                st.success("✅ All models cleared!")
                st.rerun()
    
    # Main tabs
    tab1, tab2 = st.tabs(["🎒 LV Customer Service", "📊 Schema-Action Query"])
    
    with tab1:
        rag_chatbot_tab()
    
    with tab2:
        schema_action_tab()


if __name__ == "__main__":
    main()