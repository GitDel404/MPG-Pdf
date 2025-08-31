from langchain_core.documents import Document
from pypdf import PdfReader



def document_uploader(st):
    uploaded_files = st.file_uploader("Upload", accept_multiple_files=True, type='pdf')
    number_of_uploaded_files= len(uploaded_files)
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            
            
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
            
            # Create a LangChain Document for each PDF
                doc = Document(
                    page_content=page_text,
                    metadata={"source": uploaded_file.name}
                )
                documents.append(doc)
    return documents



