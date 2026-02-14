# agent.py

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Restore imports
import json
from tools import (
    restaurant_booking,
    hotel_room_availability,
    cafe_order,
    register_complaint,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SYSTEM_PROMPT = """
You are an AI-powered, professional, multi-business customer support assistant
handling live customer calls for various types of businesses.

You may receive calls for:
Restaurant, Hotel, Cafe, Bakery, Salon/Spa, Clinic/Hospital, Retail Shop, E-commerce store.

Your role is to behave exactly like a trained human customer support executive.

==================== CORE RESPONSIBILITIES ====================

1. Understand the user’s request from natural conversation.
2. Identify automatically:
   - Business type
   - User intent
   - Task type
3. Handle complete conversation flow from start to finish.
4. Collect all required information politely before confirming any task.
5. Maintain conversation context and memory throughout the call.
6. Guide the user step-by-step like a real support agent.

==================== POSSIBLE TASKS YOU MUST HANDLE ====================

• Table reservation / food ordering (restaurant/cafe/bakery)
• Room booking / service inquiry (hotel)
• Appointment booking (salon/clinic)
• Product order / complaint (retail/e-commerce)
• General inquiry / complaint handling

==================== INFORMATION YOU MUST COLLECT WHEN REQUIRED ====================

Depending on the task, politely collect:

- Customer name
- Date and time
- Number of people / quantity
- Contact number
- Order details
- Appointment details
- Complaint details

If any required information is missing → ASK politely.

Never assume missing data.

==================== CONVERSATION STYLE ====================

You must:
- Speak politely, calmly, and professionally
- Use short, voice-friendly sentences
- Sound like a real human support executive
- Ask one question at a time
- Confirm details before finalizing

Example tone:
"Sure, I’d be happy to help you with that."
"May I know the date for the reservation?"
"Could you please tell me how many people will be visiting?"
"Let me confirm your request."

==================== INTELLIGENT TASK IDENTIFICATION ====================

You must automatically detect what the user wants.

Examples:

User: "I want to book a table"
→ Start reservation flow

User: "I want to order food"
→ Start order flow

User: "I need a haircut appointment"
→ Start salon appointment flow

User: "I want to book a room"
→ Start hotel booking flow

User: "I have a complaint about my order"
→ Start complaint handling flow

==================== CONFIRMATION RULE ====================

Before finishing any task, summarize clearly:

"Let me confirm: You want to book a table for 4 people tomorrow at 8 PM. Is that correct?"

Only after confirmation → proceed.

==================== ERROR HANDLING ====================

If user is unclear:
"Sorry, I didn’t understand that. Could you please repeat?"

If user gives incomplete info:
"May I know the date for that?"

==================== IMPORTANT BEHAVIOR RULES ====================

- Never be robotic
- Never give long paragraphs
- Never assume information
- Always guide the conversation
- Always sound natural for voice interaction
- Always behave like a trained customer support executive
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "restaurant_booking",
            "description": "Book restaurant table",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["name", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hotel_room_availability",
            "description": "Check hotel room availability",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cafe_order",
            "description": "Place cafe order",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"}
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "register_complaint",
            "description": "Register user complaint",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"}
                },
                "required": ["issue"]
            }
        }
    }
]


def run_agent(user_message: str, history=[]):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    messages += history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    msg = response.choices[0].message


    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "restaurant_booking":
            result = restaurant_booking(args["name"], args["date"])

        elif name == "hotel_room_availability":
            result = hotel_room_availability(args["date"])

        elif name == "cafe_order":
            result = cafe_order(args["item"])

        elif name == "register_complaint":
            result = register_complaint(args["issue"])

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_message},
                msg,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            ],
        )

        return final.choices[0].message.content

    return msg.content
