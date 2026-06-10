import os
import discord
from dotenv import load_dotenv

from moderation import check_moderation
from rag_engine import get_answer, initialize_rag

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not found in environment variables. Please check your .env file.")

# Set up Discord Intents
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    # Initialize RAG on startup so it's ready
    initialize_rag()
    print('RAG system initialized and ready.')

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # 1. Moderation Check (Real-time filtering of toxic behavior)
    # We run this check before processing the message
    is_toxic = check_moderation(message.content)
    
    if is_toxic:
        print(f"Toxic message flagged from {message.author}: {message.content}")
        # Delete the toxic message
        try:
            await message.delete()
        except discord.errors.Forbidden:
            print("Bot lacks permissions to delete messages.")
        
        # Warn the user
        await message.channel.send(f"{message.author.mention}, your message was flagged for violating community guidelines and was removed.")
        return

    # 2. RAG & Conversation processing
    # Indicate that the bot is typing while processing
    async with message.channel.typing():
        try:
            # We use the user's ID as the session ID for context caching in SQLite
            user_session_id = str(message.author.id)
            
            # Get the answer from LangChain RAG engine
            answer = get_answer(user_session_id, message.content)
            
            # Send the response back to the channel
            await message.channel.send(answer)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            await message.channel.send("Sorry, I encountered an error while trying to process your request.")

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
