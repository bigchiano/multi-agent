use async_openai::{
    config::OpenAIConfig,
    types::{CreateChatCompletionRequestArgs, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs},
    Client,
};
use std::error::Error;

const OLLAMA_MODEL: &str = "llama3.2";

/// Helper function to handle boilerplate OpenAI API calls asynchronously.
async fn call_llm(client: &Client<OpenAIConfig>, system_prompt: &str, user_input: &str) -> Result<String, Box<dyn Error>> {
    let request = CreateChatCompletionRequestArgs::default()
        .model(OLLAMA_MODEL)
        .temperature(0.5)
        .messages([
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system_prompt)
                .build()?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_input)
                .build()?
                .into(),
        ])
        .build()?;

    let response = client.chat().create(request).await?;
    
    // Extract text content from the first choice response
    let choice = response.choices.first()
        .ok_or("No response choices returned from LLM")?;
    
    let content = choice.message.content.clone()
        .ok_or("LLM returned an empty message payload")?;

    Ok(content)
}

/// Creates a structured, logical 3-point outline for the topic.
async fn planner_agent(client: &Client<OpenAIConfig>, topic: &str) -> Result<String, Box<dyn Error>> {
    let system_prompt = "You are an expert Planner Agent. Your job is to create a rigid, \
                         structured 3-point markdown outline for any topic given to you. \
                         Do not write full paragraphs, only the structural outline.";
    
    let user_input = format!("Create an outline for: {}", topic);
    call_llm(client, system_prompt, &user_input).await
}

/// Transforms a raw outline into comprehensive, highly educational text.
async fn writer_agent(client: &Client<OpenAIConfig>, outline: &str) -> Result<String, Box<dyn Error>> {
    let system_prompt = "You are an expert Writer Agent. Take the provided structural outline \
                         and expand it into clear, beginner-friendly, and educational paragraphs. \
                         Maintain the structural layout using clean markdown headers.";
    
    let user_input = format!("Draft content based on this outline:\n{}", outline);
    call_llm(client, system_prompt, &user_input).await
}

/// Reviews text for clarity, grammatical issues, and professional presentation.
async fn verifier_agent(client: &Client<OpenAIConfig>, draft: &str) -> Result<String, Box<dyn Error>> {
    let system_prompt = "You are a meticulous Verifier Agent. Review the provided draft. \
                         Correct any awkward phrasing, enhance readability, remove fluff, and ensure \
                         the markdown spacing looks flawless. Return only the polished final text.";
    
    let user_input = format!("Review and polish this draft:\n{}", draft);
    call_llm(client, system_prompt, &user_input).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. Point the OpenAI-compatible client at a local Ollama server instead of api.openai.com.
    //    Ollama ignores the API key but async-openai requires a non-empty value.
    let config = OpenAIConfig::new()
        .with_api_base("http://localhost:11434/v1")
        .with_api_key("ollama");
    let client = Client::with_config(config);
    let target_topic = "How quantum computing impacts modern data encryption";

    println!("🚀 Launching Rust multi-agent pipeline for topic: '{}'...\n", target_topic);

    // Step 1: Planning Phase
    println!("🤖 [Planner Agent]: Generating structural blueprint...");
    let outline = planner_agent(&client, target_topic).await?;
    println!("--- Outline Generated successfully. ---\n");

    // Step 2: Writing Phase (Handoff 1)
    println!("🤖 [Writer Agent]: Expanding blueprint into complete paragraphs...");
    let draft = writer_agent(&client, &outline).await?;
    println!("--- Draft Written successfully. ---\n");

    // Step 3: Verification Phase (Handoff 2)
    println!("🤖 [Verifier Agent]: Polishing text and checking structural quality...");
    let final_output = verifier_agent(&client, &draft).await?;
    println!("✨ --- Pipeline Complete! Final Output below: --- ✨\n");

    println!("{}", final_output);

    Ok(())
}
