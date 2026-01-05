try:
    import pwd
    print("pwd available")
except ImportError:
    print("pwd NOT available")

try:
    from langchain_community.document_loaders.pdf import PyPDFLoader
    print("PyPDFLoader OK")
except Exception as e:
    print(f"PyPDFLoader error: {e}")

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    print("langchain_community.document_loaders OK")
except Exception as e:
    print(f"langchain_community.document_loaders error: {e}")
