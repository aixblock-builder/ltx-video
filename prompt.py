# from langchain.chains import LLMChain
import re

from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline


def qa_with_context(model, context, question):
    # Create HuggingFacePipeline from the model
    hf = HuggingFacePipeline(pipeline=model)
    print("context:", context)
    print("question:", question)
    # Context text that the model will use to answer the question
    # context = """
    # Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.
    # """

    # Create a prompt template for QA with context
    template = f"""
    Here is the context: 
    {context}

    Based on the above context, provide an answer to the following question: 
    {question}

    Answer:
    """

    final_answer = ""
    try:
        # Create a prompt from the template, using context and question
        # qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # # Create an LLMChain with Llama for contextual QA
        # qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

        # # Run the QA chain with context and question
        # result = qa_chain.run({"context": context, "question": question})
        # print(result)
        prompt = PromptTemplate.from_template(template)

        chain = prompt | hf
        result = chain.invoke({"question": question, "context": context})

        # Run the QA chain with context and question
        # result = qa_chain.run({"context": context, "question": question})
        # print(result)

        answer = re.search(r"Answer:\s*(.*)", result)
        # Kiểm tra và lấy phần trả lời, loại bỏ khoảng trắng và dấu chấm
        if answer:
            final_answer = re.sub(
                r"[^\w\s]", "", answer.group(1)
            ).strip()  # Loại bỏ dấu chấm ở cuối nếu có
            print(final_answer)
        else:
            print("No answer found.")
    except Exception as e:
        print(e)

    return final_answer


def qa_without_context(model, question):
    # Create HuggingFacePipeline from the model
    hf = HuggingFacePipeline(pipeline=model)
    print("question")
    # Context text that the model will use to answer the question
    # context = """
    # Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.
    # """

    # Create a prompt template for QA with context
    template = f"""
    Please answer the following question to the best of your ability:
    Question: 
    {question}
    
    Answer:
    """

    final_answer = ""
    try:

        # Tạo PromptTemplate chỉ với câu hỏi
        # qa_no_context_prompt_template = PromptTemplate(template=qa_no_context_prompt, input_variables=["question"])

        # # Tạo LLMChain cho QA không cần ngữ cảnh
        # qa_no_context_chain = LLMChain(llm=llm, prompt=qa_no_context_prompt_template)
        prompt = PromptTemplate.from_template(template)

        chain = prompt | hf
        result = chain.invoke({"question": question})

        # Chạy mô hình để trả lời câu hỏi
        # result = qa_no_context_chain.run({"question": question})

        answer = re.search(r"Answer:\s*(.*)", result)
        # Kiểm tra và lấy phần trả lời, loại bỏ khoảng trắng và dấu chấm
        if answer:
            final_answer = re.sub(
                r"[^\w\s]", "", answer.group(1)
            ).strip()  # Loại bỏ dấu chấm ở cuối nếu có
            print(final_answer)
        else:
            print("No answer found.")
    except Exception as e:
        print(e)

    return final_answer


def text_classification(model, context, categories):
    hf = HuggingFacePipeline(pipeline=model)

    # Tạo prompt cho task phân loại văn bản
    template = f"""Classify the following text into one of the following categories:
    {categories}

    The text is: {context}

    Classification:
    """

    final_answer = ""
    try:
        # Tạo prompt từ template, sử dụng văn bản cần phân loại
        # classification_prompt_template = PromptTemplate(template=classification_prompt, input_variables=["context"])

        # # Tạo LLMChain cho task phân loại văn bản
        # classification_chain = LLMChain(llm=llm, prompt=classification_prompt_template)

        # # Chạy mô hình với văn bản cần phân loại
        # result = classification_chain.run({"context": context})
        prompt = PromptTemplate.from_template(template)

        chain = prompt | hf
        result = chain.invoke({"context": context, "categories": categories})

        # Sử dụng regex để trích xuất phần Classification
        classification = re.search(r"Classification:\s*(.*)", result)

        if classification:
            final_answer = re.sub(r"[^\w\s]", "", classification.group(1)).strip()
            print("Classification:", final_answer)
        else:
            print("No classification found.")

    except Exception as e:
        print(e)

    return final_answer


def text_summarization(model, context):
    hf = HuggingFacePipeline(pipeline=model)

    # Tạo prompt cho task tóm tắt văn bản
    template = f"""
    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

    Text: 
    {context}

    Summary:
    """

    final_summary = ""
    try:
        # Tạo prompt từ template, sử dụng văn bản cần tóm tắt
        # summary_prompt_template = PromptTemplate(template=summary_prompt, input_variables=["context"])

        # # Tạo LLMChain cho task tóm tắt văn bản
        # summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template)

        # # Chạy mô hình với văn bản cần tóm tắt
        # result = summary_chain.run({"context": context})

        prompt = PromptTemplate.from_template(template)

        chain = prompt | hf

        result = chain.invoke({"context": context})
        print(result)

        # Sử dụng regex để trích xuất phần Summary
        summary = re.search(r"Summary:\s*(.+)", result, re.DOTALL)

        if summary:
            final_summary = re.sub(r"[^\w\s.,!?]", "", summary.group(1)).strip()
            print("Summary:", final_summary)
        else:
            print("No summary found.")

    except Exception as e:
        print(e)

    return final_summary


def text_ner(model, context, categories):
    hf = HuggingFacePipeline(pipeline=model)

    # Tạo prompt cho task NER
    template = f"""
    The text is: {context}

    Extract all named entities from the context and classify them into the categories:
    {categories}

    Named Entities-classification:
    """

    final_entities = ""
    try:
        # Tạo prompt từ template, sử dụng văn bản cần phân tích NER
        # ner_prompt_template = PromptTemplate(template=ner_prompt, input_variables=["context"])

        # # Tạo LLMChain cho task NER
        # ner_chain = LLMChain(llm=llm, prompt=ner_prompt_template)

        # # Chạy mô hình với văn bản cần phân tích
        # result = ner_chain.run({"context": context})

        # print("Raw Result:\n", result)
        prompt = PromptTemplate.from_template(template)

        chain = prompt | hf

        result = chain.invoke({"context": context, "categories": categories})
        print(result)

        # Trích xuất phần "Named Entities-classification:" và parse các NER
        ner_classification = re.search(
            r"Named Entities-classification:\s*(.*)", result, re.DOTALL
        )

        if ner_classification:
            # Lấy danh sách các entity từ kết quả, chia theo dòng
            final_entities = ner_classification.group(1).strip()

            # entities = entities_text.split("\n")

            # # Duyệt qua các entity và chuyển đổi thành format mong muốn
            # for entity in entities:
            #     match = re.match(r"\d+\.\s*(\w+):\s*(.*)", entity.strip())
            #     if match:
            #         entity_type = match.group(1).upper()  # Loại entity (Person, Location, Organization)
            #         entity_value = match.group(2).strip()  # Giá trị entity

            #         # Kiểm tra nếu value có nhiều địa điểm, tách ra
            #         if entity_type == 'LOCATION' and ',' in entity_value:
            #             # Tách value nếu chứa dấu phẩy
            #             location_values = [val.strip() for val in entity_value.split(',')]
            #             # Thêm từng phần vào final_entities dưới dạng các đối tượng riêng biệt
            #             for location in location_values:
            #                 final_entities.append({"type": "LOCATION", "value": location})
            #         else:
            #             # Thêm entity vào danh sách nếu không phải LOCATION hoặc không có dấu phẩy
            #             final_entities.append({"type": entity_type, "value": entity_value})

    except Exception as e:
        print(f"Error: {e}")

    return final_entities
