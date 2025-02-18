import os
import streamlit as st
from langchain.schema import Document
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key=os.environ["HF_TOKEN"]

model_name = "sentence-transformers/all-MiniLM-L6-v2"

examples = [
    {
        "input": "Give me records of items stuck in carton error",
        "query": "SELECT order_number, status, error_message, t.imported_date, order_type, total_lines, total_units, * FROM t_order t (NOLOCK) WHERE status = 'CRTNERROR' ORDER BY t.imported_date DESC"
    },
    {
        "input": "Orders with no t_pick_detail_cartonize record",
        "query": "SELECT ord.order_number, ord.status, ord.host_status, pdc.status, pdc.order_number, imported_date FROM t_order ord (NOLOCK) LEFT JOIN t_pick_detail_cartonize pdc (NOLOCK) ON ord.order_number = pdc.order_number WHERE imported_date < DATEADD(mi, -5, GETDATE()) AND pdc.order_number IS NULL"
    },
    {
        "input": "Give me records where picked inventory is zero",
        "query": "SELECT * FROM t_stored_item sto (NOLOCK) WHERE actual_qty = 0 AND type <> 0"
    },
    {
        "input": "Give me details for picking status, container ID (LPN) for the order pick line (line_number=1), pull lines (line_number!=1) for the order number or packwaves",
        "query": "SELECT TOP 2 pick_id, order_number, line_number, work_type, status, item_number, planned_quantity, picked_quantity, staged_quantity, pick_location, wave_id, pack_wave_id, load_id, container_id, user_assigned, wh_id, created_date, pick_location_change_date, prev_pick_id, priority, picked_date, * FROM t_pick_detail (NOLOCK) WHERE order_number IN ('')"
    },
    {
        "input": "Give me LPNs details which are of type inventory (IV), loaded (LO), replen (RP), staged order (SO), wave pulls (WP) or LPN orders associated with orders (control_number) or LPNs in status available (A), HOLD (H) or pallet number (parent_hu_id) or label_status in (NULL, COMPLETE, ERROR, PANDA, PREP, PROCESSED)",
        "query": "SELECT * FROM t_hu_master WHERE type IN ('IV', 'LO', 'RP', 'SO', 'WP') OR control_number IS NOT NULL OR status IN ('A', 'H') OR parent_hu_id IS NOT NULL OR label_status IN (NULL, 'COMPLETE', 'ERROR', 'PANDA', 'PREP', 'PROCESSED')"
    },
    {
        "input": "Give me employee details, user is logged into device if device_id is not NULL",
        "query": "SELECT id, name, emp_number, dept, supervisor, menu_level, wh_id, hu_id, device, change_pick_type_allowed, picking_type, can_clear_pallet_positions_flag, chute_packout_full_scan, allow_move_hold_items, login_time, logout_time, last_activity, role, email, phone_number, address, * FROM t_employee WHERE device_id IS NOT NULL"
    },
    {
        "input": "Please provide me details of items stored. If type=0, it's not allocated for any orders. If type is other than (0, -99), it has a foreign reference to t_pick_detail with t_stored_item.type=t_pick_detail.pick",
        "query": "SELECT item_number, actual_qty, unavailable_qty, status, wh_id, location_id, expiration_date, lot_number, inspection_code, serial_number, type, put_away_location, stored_attribute_id, hu_id, shipment_number, item_name, supplier_id, batch_number, created_date, updated_date, * FROM t_stored_item WHERE (type = 0 OR type NOT IN (0, -99)) AND EXISTS (SELECT 1 FROM t_pick_detail WHERE t_stored_item.type = t_pick_detail.pick)"
    },
    {
        "input": "Give me user details, it has device_id (users logged on)",
        "query": "SELECT user_id, user_name, email, phone_number, role, department, supervisor_id, login_time, logout_time, last_activity, device_id, device_type, employee_id, fork_id, wh_id, business_process, text, last_tran_datetime, last_process, last_message_id, last_response_json, * FROM t_user WHERE device_id IS NOT NULL"
    },
    {
        "input": "Give me LPNs which are on hold",
        "query": "SELECT * FROM t_hu_master (NOLOCK) WHERE hu_id IN (SELECT container_number FROM t_pick_detail (NOLOCK) WHERE order_number IN ('')) AND status NOT IN ('A')"
    }
]



# Create an in-memory docstore
docstore = InMemoryDocstore({i: Document(page_content=example["input"], metadata={"query": example["query"]}) for i, example in enumerate(examples)})
index_to_docstore_id = {i: i for i in range(len(examples))}

# Create an in-memory docstore
docstore = InMemoryDocstore({i: Document(page_content=example["input"], metadata={"query": example["query"]}) for i, example in enumerate(examples)})
index_to_docstore_id = {i: i for i in range(len(examples))}

@st.cache_resource
def get_example_selector():
    # Initialize FAISS embeddings
    hf_embeddings = HuggingFaceEndpointEmbeddings()
    dim=384
    # Create a FAISS index
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(embedding_function=hf_embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        hf_embeddings,
        vectorstore,
        k=3,
        input_keys=["input"],  
    )
    return example_selector

#example_selector = get_example_selector()



# # Example usage in a Streamlit app
# st.write("Example selector initialized with FAISS vector store")

# # Provide input to get relevant examples
# user_input = st.text_input("Enter your query")
# if user_input:
#     selected_examples = example_selector.get(user_input)
#     st.write("Relevant examples:")
#     st.write(selected_examples)

