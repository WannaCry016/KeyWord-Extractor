# Explanation of Technical Approach

So I have tried 3 approaches to see which worked the best with the required latency and I will quickly explain all the 3 one by one.

---

## `a1.py`

So In this approach I tried to extract all keywords using spacy NLP model for Named Entity Recognition - this extracted all the noun chunks, named entities (product, organisations) etc.  
Then we have to keep the keywords up to the context of commercial intent so for that I used sentence embeddings of a small but fast llm which converted the conversation into embeddings and then did the matching using cosine similarity to get the keywords that best suite to the context and removed the Similar or overlapping phrases are filtered using fuzzy matching.  
And for the Commercial intent prediction we will need a trained model like xgboost or random forest would work and a good dataset if we want to do it fast and in ml way. As currently there is no such dataset so for now i have just implemented a simple method to just search for some keywords to score Thats it.  
For Product Categories I have made a list of variety of product sectors and then i mapped them to the primary topics and retrieve the top-k categories by applying a threshold. In this case also we could use a ml model but there is no current dataset.

![Screenshot 2025-05-22 133555](https://github.com/user-attachments/assets/ee474d19-9644-4696-a309-e493d4eeee1d)

So this has potential to work in very less latency if the commercial intent and product categorie prediction model is built.

---

## `a2.py`

In this approach I used RAG method so I extracted the keyword normally through the 1st approach and gave it to a llm using a cloud api (I have used groq) with the other task to predict the commercial intent and product categories.  
This worked great with respect to the product categories but commercial intent was not that relevant as it was not able to get the full intent by seeing the keywords only and was giving avg latency of around 500 ms which ig could be improved if worked with entripise level api and good network conditions.

![Screenshot 2025-05-22 134604](https://github.com/user-attachments/assets/1a690c62-6082-4bf1-b7ff-3ff623ee5407)

---

## `a3.py`

In this I completely gave the converstation and the task to the llm using api and It was working great in keyword , intent scoring and product category with avg latency around 500 which could be definitely improved with enterprise lvl api.

![Screenshot 2025-05-22 134855](https://github.com/user-attachments/assets/66946341-1c8d-43ed-b650-8f3fe683cac3)

---

## Conclusion

So to conclude the 3rd one seems to be the best approach till now as we need a model which understands the context then only we will be able to score and get the keywords well from there. It is showing little more latency but it can work in under 200ms as I was using free api so its queued in public stack.
