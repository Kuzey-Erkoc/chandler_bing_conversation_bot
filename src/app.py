import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
from main import main, Conversation, load_saved_model
from conversation import output_dir, train_df, validation_df

def run_script(use_saved_model=False):
    try:
        if use_saved_model and os.path.exists(os.path.join(output_dir, "config.json")):
            print(f"Loading saved model from {output_dir}...")
            model, tokenizer = load_saved_model(output_dir)
        else:
            if use_saved_model:
                print(f"Saved model not found in {output_dir}. Proceeding with training...")
            else:
                print("Training new model...")
            print("Running main script...")
            main(train_df, validation_df)
            print("Loading newly trained model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            model = AutoModelForCausalLM.from_pretrained(output_dir)

        conversation = Conversation(model, tokenizer)

        def chatbot(message):
            response = conversation.chat(message)
            return response

        print("\nLaunching Gradio interface...")
        iface = gr.Interface(
            fn=chatbot,
            inputs=gr.Textbox(lines=2, placeholder="Type your message here...", label="Your Message"),
            outputs=gr.Textbox(label="Response"),
            title="Chandler Bing Chatbot",
            description="Chat with a bot trained to talk like Chandler Bing from Friends!",
            examples=[["How you doin'?"], ["Could I BE any more sarcastic?"], ["What's your job?"]],
            theme="compact"
        )

        iface.launch(share=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Falling back to training a new model...")
        main(train_df, validation_df)
        print("Loading newly trained model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        # Now proceed with the chatbot setup as before
        conversation = Conversation(model, tokenizer)
        # ... (rest of the Gradio setup)

if __name__ == "__main__":
    use_saved_model = True
    run_script(use_saved_model)