# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function
from reasoning_agent import create_reasoning_compatible_agent

_BASE_SYSTEM_MSG = (
    "You are a helpful travel planning assistant. Always be professional and provide accurate information."
)

#from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
#from semantic_kernel.agents.chat_completion.reasoning_compatible_agent import ReasoningCompatibleAgent
from semantic_kernel.contents import AuthorRole, ChatMessageContent

class FlightPlugin:
    @kernel_function
    def book_flight(self, flight_id: str) -> str:
        """Book a specific flight."""
        return f"Successfully booked flight with ID {flight_id}. Your booking reference is FLX12345."


class HotelPlugin:
    @kernel_function
    def book_hotel(self, hotel_id: str) -> str:
        """Book a specific hotel."""
        return f"Successfully booked hotel with ID {hotel_id}. Your booking reference is HTX12345."


class PlanningPlugin:
    @kernel_function
    def get_weather(self, location: str) -> str:
        """Get weather information for a location."""
        return f"Weather information for {location}: Sunny, 25°C."

    @kernel_function
    def search_hotels(self, location: str, check_in: str, check_out: str) -> str:
        """Search for available hotels."""
        available_hotels = [
            {"hotel_id": "HT123", "name": "Hotel Sunshine", "price": "$150/night", "accommodates": "2 people"},
            {"hotel_id": "HT456", "name": "Ocean View Resort", "price": "$200/night", "accommodates": "4 people"},
            {"hotel_id": "HT789", "name": "Mountain Retreat", "price": "$180/night", "accommodates": "2 people"},
        ]
        return f"Searching hotels in {location} from {check_in} to {check_out}:\n{available_hotels}"

    @kernel_function
    def search_flights(self, origin: str, destination: str, date: str) -> str:
        """Search for available flights."""
        available_flights = [
            {"flight_id": "FL123", "take-off-time": "10:00 AM", "arrival-time": "12:00 PM", "price": "$200"},
            {"flight_id": "FL456", "take-off-time": "2:00 PM", "arrival-time": "4:00 PM", "price": "$250"},
            {"flight_id": "FL789", "take-off-time": "6:00 PM", "arrival-time": "8:00 PM", "price": "$300"},
        ]
        return f"Available flights from {origin} to {destination} on {date}:\n{available_flights}"


def get_agents() -> dict[str, ChatCompletionAgent]:
    """Creates and returns a set of agents for the travel planning system."""
    print("Using ReasoningCompatibleAgent (auto-detects standard vs reasoning models)")
    
    # Create reasoning-compatible agents that auto-detect model type
    conversation_manager = create_reasoning_compatible_agent(
        name="conversation_manager",
        description="Manages conversation flow and coordinates between agents",
        instructions="You are a conversation manager for a travel planning system. "
                    "Coordinate between different agents to help users plan their trips.",
    )

    planner = create_reasoning_compatible_agent(
        name="planner",
        description="Creates comprehensive travel plans including flights, hotels, and activities",
        instructions="You are a travel planner. Create detailed travel itineraries "
                    "including flights, hotels, and activities based on user preferences.",
        plugins=[PlanningPlugin()],
    )

    router = create_reasoning_compatible_agent(
        name="router",
        description="Routes tasks to appropriate specialized agents",
        instructions="You are a router agent. Analyze user requests and direct them "
                    "to the most appropriate specialist agent.",
    )

    destination_expert = create_reasoning_compatible_agent(
        name="destination_expert",
        description="Expert in destination recommendations and local information",
        instructions="You are a destination expert. Provide detailed information about "
                    "travel destinations, local attractions, and travel tips.",
        plugins=[PlanningPlugin()],
    )

    flight_agent = create_reasoning_compatible_agent(
        name="flight_agent",
        description="Specializes in flight booking",
        instructions="You are a flight booking specialist. Help users search for "
                    "and book flights that meet their travel needs.",
        plugins=[FlightPlugin()],
    )

    hotel_agent = create_reasoning_compatible_agent(
        name="hotel_agent",
        description="Specializes in hotel booking",
        instructions="You are a hotel booking specialist. Help users search for "
                    "and book hotels that meet their accommodation needs.",
        plugins=[HotelPlugin()],
    )

    return {
        conversation_manager.name: conversation_manager,
        planner.name: planner,
        router.name: router,
        destination_expert.name: destination_expert,
        flight_agent.name: flight_agent,
        hotel_agent.name: hotel_agent,
    }