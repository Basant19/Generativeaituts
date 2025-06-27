from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=r'D:\Generative_ai_practise\Basic_model_setup\rags\documentloaders\Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])