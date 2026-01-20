import logging
import json
import re
import datetime
from typing import Dict, List, Optional, Any
from .tool_core import BaseTool, ToolResult, ToolType, LLMService

# Configure logger
logger = logging.getLogger(__name__)

class FitnessPlanGeneratorTool(BaseTool):
    def __init__(self, llm_service: LLMService, api_key: str, endpoint_url: str):
        super().__init__(
            ToolType.FITNESS_PLAN_GENERATOR.value,
            "Generates a personalized weekly workout plan based on user profile and goals."
        )
        self.llm_service = llm_service
        self.api_key = api_key
        self.endpoint_url = endpoint_url

    async def execute(self, query: str, user_profile: Dict, **kwargs) -> ToolResult:
        """
        Executes the fitness plan generation logic.
        """
        try:
            logger.info(f"Generating fitness plan for user profile: {user_profile}")

            # 1. Validate Input
            required_fields = ['height_cm', 'primary_goal', 'fitness_level', 'days_per_week', 'available_equipment']
            missing = [f for f in required_fields if not user_profile.get(f)]
            
            if missing:
                return ToolResult(
                    success=False, 
                    error=f"Missing required fitness profile data: {', '.join(missing)}. Please update your profile first."
                )

            # 2. Construct System Prompt
            system_prompt = f"""
            You are an elite Personal Trainer and Physiotherapist. Create a highly personalized Weekly Workout Plan.
            
            **User Profile:**
            - **Goal:** {user_profile.get('primary_goal')}
            - **Fitness Level:** {user_profile.get('fitness_level')}
            - **Height:** {user_profile.get('height_cm')} cm
            - **Weight:** {user_profile.get('weight_kg', 'N/A')} kg
            - **Age:** {user_profile.get('age', 'N/A')}
            - **Gender:** {user_profile.get('gender', 'N/A')}
            - **Injuries/Conditions:** {', '.join(user_profile.get('medical_conditions', [])) or 'None'}
            - **Days Available:** {', '.join(user_profile.get('days_per_week', []))}
            - **Equipment:** {', '.join(user_profile.get('available_equipment', []))}

            **MANDATORY OUTPUT FORMAT (JSON ONLY):**
            You must return a valid JSON object with this exact structure:
            {{
                "weekly_schedule": {{
                    "Monday": {{ "focus": "...", "exercises": [ {{ "name": "...", "sets": "...", "reps": "...", "rest": "..." }} ] }},
                    "Tuesday": ... (include Rest days if applicable)
                }},
                "warmup_routine": [ "step 1", "step 2" ],
                "cool_down_routine": [ "step 1", "step 2" ],
                "nutrition_tips": [ "tip 1", "tip 2" ],
                "progression_guide": "How to increase intensity over 4 weeks."
            }}
            
            **RULES:**
            1. Tailor volume (sets/reps) strictly to '{user_profile.get('fitness_level')}'.
            2. ONLY use equipment listed: {', '.join(user_profile.get('available_equipment', []))}.
            3. Consider medical conditions: {', '.join(user_profile.get('medical_conditions', []))}.
            4. Do NOT output markdown text like ```json ... ```. Just the raw JSON string.
            """

            # 3. Call LLM
            user_prompt = f"Generate the workout plan for: {query}"
            
            llm_response = await self.llm_service.query(
                prompt=user_prompt,
                system_prompt=system_prompt,
                chat_history=[],
                max_tokens=2500,
                temperature=0.5
            )

            # 4. Parse JSON
            try:
                # Clean potential markdown wrappers
                json_str = llm_response.replace("```json", "").replace("```", "").strip()
                plan_json = json.loads(json_str)
                
                # Convert JSON to a readable text format for the user
                formatted_plan = self._format_plan_to_text(plan_json)
                
                return ToolResult(
                    success=True, 
                    data={
                        "answer": formatted_plan,
                        "fitness_plan_json": plan_json
                    }
                )

            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {llm_response}")
                return ToolResult(success=False, error="I generated a plan but it wasn't in the correct format. Please try again.")

        except Exception as e:
            logger.error(f"Error in FitnessPlanGeneratorTool: {e}")
            return ToolResult(success=False, error=f"An error occurred while generating your fitness plan: {str(e)}")

    def _format_plan_to_text(self, plan_json: Dict) -> str:
        """Converts the JSON plan into a nice readable string."""
        output = ["**🏋️ Your Personalized Weekly Workout Plan**\n"]
        
        schedule = plan_json.get("weekly_schedule", {})
        for day, routine in schedule.items():
            if isinstance(routine, str): # Handle "Rest Day" strings
                output.append(f"**{day}:** {routine}")
                continue
                
            focus = routine.get("focus", "General")
            output.append(f"\n**{day} - {focus}**")
            
            exercises = routine.get("exercises", [])
            if not exercises:
                output.append("  *Rest or Active Recovery*")
            else:
                for ex in exercises:
                    output.append(f"  - **{ex.get('name')}**: {ex.get('sets')} sets x {ex.get('reps')} reps ({ex.get('rest', '60s')} rest)")

        output.append("\n**🔥 Warm-up:**")
        for step in plan_json.get("warmup_routine", []):
            output.append(f"- {step}")

        output.append("\n**❄️ Cool-down:**")
        for step in plan_json.get("cool_down_routine", []):
            output.append(f"- {step}")

        return "\n".join(output)


class WorkoutAdjusterTool(BaseTool):
    def __init__(self, llm_service: LLMService):
        super().__init__(
            ToolType.WORKOUT_PLAN_ADJUSTER.value,
            "Modifies an existing workout plan based on user feedback (e.g., make it harder, change days)."
        )
        self.llm_service = llm_service

    async def execute(self, query: str, last_agent_context: Dict, **kwargs) -> ToolResult:
        """
        Executes the workout adjustment logic.
        """
        try:
            # 1. Retrieve Active Plan
            active_plan = last_agent_context.get("active_fitness_plan_json")
            if not active_plan:
                return ToolResult(success=False, error="I can't modify your plan because I don't have an active workout plan in memory yet. Please ask me to generate one first!")

            logger.info(f"Modifying workout plan with query: {query}")

            # 2. Construct System Prompt
            system_prompt = f"""
            You are an expert fitness coach. Your task is to MODIFY the existing workout plan JSON based on the user's feedback.

            **Current Plan JSON:**
            {json.dumps(active_plan)}

            **User Feedback:** "{query}"

            **Instructions:**
            1. Apply the changes strictly to the JSON structure.
            2. If the user says "make it harder", increase sets/reps or change exercises.
            3. If the user says "I don't have a bench", replace bench exercises.
            4. Keep the rest of the plan consistent.
            5. Return ONLY the updated JSON object. No markdown.
            """

            # 3. Call LLM
            llm_response = await self.llm_service.query(
                prompt=query,
                system_prompt=system_prompt,
                chat_history=[],
                max_tokens=2500,
                temperature=0.5
            )

            # 4. Parse JSON
            try:
                json_str = llm_response.replace("```json", "").replace("```", "").strip()
                updated_plan_json = json.loads(json_str)
                
                # Format to text
                # We can reuse the formatter from the other class if we make it static or duplicate it.
                # For simplicity here, I'll duplicate the simple formatter logic.
                formatted_plan = self._format_plan_to_text(updated_plan_json)

                return ToolResult(
                    success=True,
                    data={
                        "answer": f"I've updated your workout plan based on your request:\n\n{formatted_plan}",
                        "fitness_plan_json": updated_plan_json
                    }
                )

            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response during adjustment: {llm_response}")
                return ToolResult(success=False, error="I tried to update the plan, but the format got messed up. Please try again.")

        except Exception as e:
            logger.error(f"Error in WorkoutAdjusterTool: {e}")
            return ToolResult(success=False, error=f"An error occurred while modifying your plan: {str(e)}")

    def _format_plan_to_text(self, plan_json: Dict) -> str:
        """Helper to format JSON plan to text."""
        output = ["**🔄 Updated Weekly Workout Plan**\n"]
        
        schedule = plan_json.get("weekly_schedule", {})
        for day, routine in schedule.items():
            if isinstance(routine, str):
                output.append(f"**{day}:** {routine}")
                continue
                
            focus = routine.get("focus", "General")
            output.append(f"\n**{day} - {focus}**")
            
            exercises = routine.get("exercises", [])
            for ex in exercises:
                output.append(f"  - **{ex.get('name')}**: {ex.get('sets')} sets x {ex.get('reps')} reps")
                
        return "\n".join(output)