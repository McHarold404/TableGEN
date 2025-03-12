# Example Python script for the text-to-table pipeline
from model_inference.gpt import *

def main():
    # ------------------------------
    # Load or define your raw input text here.
    # For now, we hardcode a sample text.
    input_text = (
        "Global climate change has many effects, including melting ice, heat waves, and droughts. "
        "It is caused by the enhanced greenhouse effect, which is caused by pollution, such as carbon emissions and burning coal. "
        "Solutions to global climate change include individual efforts and international resolutions."
    )
    
    # ------------------------------
    # Step 1: Segment the input text into smaller passages.
    segmentation_prompt_path = "prompts/segmentation.txt"  # path to the segmentation prompt file
    segmentation_result = ask_chatgpt(segmentation_prompt_path, input_text)
    
    # (Commented out) Preprocess the segmentation_result:
    # passages = segmentation_result.split("__NEW_PASSAGE__")
    passages = segmentation_result.split("__NEW_PASSAGE__")  # You can refine the parsing as needed.
    
    final_tables = []
    
    # Paths for the table generation and rewriting prompts
    table_generation_prompt_path = "prompts/table_generation.txt"  # table & caption generation
    rewrite_table_prompt_path = "prompts/rewrite_table.txt"          # rewrite table with citations
    # Optionally, you might have a local structure critic prompt:
    local_structure_prompt_path = "prompts/local_structure.txt"      # local structure critic
    
    # ------------------------------
    # Process each passage
    for passage in passages:
        # Step 2: Generate a table with caption from the passage.
        table_response = ask_chatgpt(table_generation_prompt_path, passage.strip())
        
        # (Optional) Extract bullet_points and table from table_response if needed.
        # For simplicity, assume table_response is structured with markers you can split on.
        # Here we simply pass the entire response to the next step.
        rewrite_input = table_response  # You may need to split into {{ bullet_points }} and {{ table }} parts.
        
        # Step 3: Rewrite the table to add a citation column.
        rewritten_table = ask_chatgpt(rewrite_table_prompt_path, rewrite_input)
        
        # (Optional) Step 4: Check local structure with the critic prompt.
        # critic_response = ask_chatgpt(local_structure_prompt_path, some_input)
        # You can process critic_response if you need to filter or modify the table.
        
        final_tables.append(rewritten_table)
    
    # ------------------------------
    # Aggregate final tables (for example, join them by a separator)
    final_output = "\n\n".join(final_tables)
    
    # (Commented out) Save the final output to a file or further process it.
    # with open("final_table_output.txt", "w") as f:
    #     f.write(final_output)
    
    print("Final Generated Table(s):\n")
    print(final_output)

if __name__ == "__main__":
    main()
