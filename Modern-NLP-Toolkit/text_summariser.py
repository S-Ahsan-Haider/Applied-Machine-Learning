from transformers import pipeline

def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    # Load summarization pipeline and specify the GPU device
    # device=0 tells it to use your first GPU
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    
    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']


if __name__ == "__main__":
    # Example long text
    article = """
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also 
    be applied to any machine that exhibits traits associated with a human mind such as 
    learning and problem-solving. AI is being increasingly used across various industries 
    including healthcare, finance, and autonomous systems.
    """

    print("Original Text:\n", article)
    print("\nSummary:\n", summarize_text(article))