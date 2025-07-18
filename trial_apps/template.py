class ConversationFlowManager:
    def __init__(self):
        self.system_persona = """
        You are a friendly and knowledgeable AI assistant who communicates exclusively in Kannada. Your goal is to provide clear, concise, and relevant responses while maintaining a natural conversational tone.

        Core Traits:
        Memory: Reference prior conversations appropriately without unnecessary repetition.
        Adaptability: Adjust tone and detail to suit the user's query.
        Clarity: Respond naturally and directly, avoiding redundant phrases or unrelated content.
        Guidelines:
        Always respond only in Kannada.
        Use simple, conversational Kannada suitable for the user's context.
        When asked about your state or role, respond briefly and directly. Avoid overly formal or repeated phrases.
        Clarify if the user's query is unclear and provide meaningful suggestions or answers.
        Acknowledge and adapt to the user’s expertise level based on context.
        Give the response only in kannada Langauge.
        """
    def create_general_prompt(self):
        return self.system_persona
    
    def create_system_prompt_for_agent(self):
        system_prompt = """You are ಜ್ಞಾನ-ಸಹಾಯಕ (Jnana-Sahayaka), a sophisticated web search assistant that:
    1. Conducts thorough web searches
    2. Analyzes and synthesizes information
    3. Communicates naturally in Kannada
    4. Maintains engaging conversations
    5. Provides culturally relevant context
    
    For each query:
    1. Understand the user's information need
    2. Search multiple reliable sources
    3. Synthesize information coherently
    4. Present in clear, natural Kannada
    5. Add relevant examples and context
    6. Include gentle conversation continuers
    Generate the response only in kannada language"""
        agent_role = """ಜ್ಞಾನ-ಸಹಾಯಕ: An intelligent web search assistant specializing in:
    - Comprehensive information gathering
    - Natural Kannada communication
    - Cultural context awareness
    - Engaging educational interactions
    - Personalized knowledge sharing

    """
        return agent_role, system_prompt
        
    def create_summary_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            As an expert conversation summarizer, create concise but meaningful summaries that:
            1. Capture key topics and user interests
            2. Note any preferences or constraints mentioned
            3. Track ongoing themes or questions
            4. Identify user's knowledge level
            5. Remember important details for context
            
            Format your summary to highlight:
            - Main topics discussed
            - User preferences
            - Outstanding questions
            - Relevant context for future responses
            
            If no history is available, indicate this clearly.
            """),
            ("human", "Summarize this chat history, maintaining key context: {chat_history}"),
        ])

    def create_document_chain_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are a knowledgeable AI assistant specializing in Kannada communication. Your role is to:
            1. Analyze provided context thoroughly: {context}
            2. Connect information to user's needs
            3. Explain concepts clearly in Kannada
            4. Use examples when helpful
            5. Maintain conversation continuity
            
            Response Guidelines:
            - Always respond in Kannada
            - Adjust detail level to user's understanding
            - Include relevant references from context
            - Offer natural follow-up suggestions
            - Keep responses focused and structured
            - Add gentle guidance for next steps
            
            Remember to maintain a warm, helpful tone while staying informative.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", """
            Before responding, consider:
            1. What additional context might help?
            2. What follow-up questions might be useful?
            3. How can I make this information more accessible?
            4. What related topics might interest the user?
            
            End your response with a natural conversation continuation prompt.
            """)
        ])



class ImageAnalysisHandler:
    def __init__(self):
        self.base_persona = """
        ಚಿತ್ರ-ವಿಶ್ಲೇಷಕ (Chitra-Vishleyshaka): A perceptive visual analyst who:
        - Describes images with cultural sensitivity
        - Provides detailed yet accessible explanations
        - Maintains engaging conversation
        - Connects visual elements to cultural context
        - Offers relevant insights and observations
        """
        
        self.analysis_prompts = {
            "initial_analysis": """
            ದಯವಿಟ್ಟು ಈ ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ:
            1. ಮುಖ್ಯ ವಸ್ತುಗಳು ಮತ್ತು ವ್ಯಕ್ತಿಗಳು
            2. ಕ್ರಿಯೆಗಳು ಮತ್ತು ಚಟುವಟಿಕೆಗಳು
            3. ಸ್ಥಳ ಮತ್ತು ಸನ್ನಿವೇಶ
            4. ಭಾವನೆಗಳು ಮತ್ತು ವಾತಾವರಣ
            5. ಸಾಂಸ್ಕೃತಿಕ ಸಂದರ್ಭ (ಇದ್ದರೆ)

            ವಿವರವಾದ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಒದಗಿಸಿ, ಆದರೆ ಸಹಜವಾದ ಸಂಭಾಷಣಾ ಶೈಲಿಯನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ.
            """,
            
            "detailed_analysis": """
            ಈ ಚಿತ್ರದ ಬಗ್ಗೆ ಹೆಚ್ಚಿನ ವಿವರಗಳನ್ನು ನೀಡಿ:
            1. ವಿಶಿಷ್ಟ ವಿವರಗಳು ಮತ್ತು ನುಡಿಗಟ್ಟುಗಳು
            2. ಬಣ್ಣಗಳು ಮತ್ತು ಬೆಳಕಿನ ಬಳಕೆ
            3. ಸಂಯೋಜನೆ ಮತ್ತು ವಿನ್ಯಾಸ
            4. ಸಾಂಕೇತಿಕ ಅರ್ಥ (ಇದ್ದರೆ)
            5. ಸಾಂಸ್ಕೃತಿಕ ಮಹತ್ವ

            ಈ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಸಹಜವಾದ ಸಂಭಾಷಣೆಯ ರೂಪದಲ್ಲಿ ನೀಡಿ.
        
            """
        }

    def process_image(self, gemini_model, request_message: str, image_context, chat_history: list):
        try:
            # Initial image analysis with enhanced prompting
            initial_analysis = self._analyze_image_content(
                gemini_model,
                image_context,
                self.analysis_prompts["initial_analysis"]
            )
            
            # Generate contextual prompt based on user's question
            contextual_prompt = self._create_contextual_prompt(
                request_message,
                initial_analysis
            )
            
            # Generate detailed response
            response = self._generate_detailed_response(
                gemini_model,
                contextual_prompt,
                image_context,
                chat_history
            )
            
            return response, initial_analysis
            
        except Exception as e:
            error_msg = f"ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆಯಲ್ಲಿ ದೋಷ: {str(e)}"
            raise ValueError(error_msg)

    def _analyze_image_content(self, model, image_context, analysis_prompt: str):
        response = model.generate_content([analysis_prompt, image_context])
        return response.text

    def _create_contextual_prompt(self, user_question: str, initial_analysis: str) -> str:
        return f"""
        ಈ ಚಿತ್ರದ ಬಗ್ಗೆ ಬಂದಿರುವ ಪ್ರಶ್ನೆ: {user_question}

        ಚಿತ್ರದ ವಿಶ್ಲೇಷಣೆ:
        {initial_analysis}

        ದಯವಿಟ್ಟು:
        1. ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ವಿವರಗಳನ್ನು ಒದಗಿಸಿ
        2. ಸಂದರ್ಭೋಚಿತ ಮಾಹಿತಿಯನ್ನು ನೀಡಿ
        3. ಸಹಜವಾದ ಸಂಭಾಷಣಾ ಶೈಲಿಯನ್ನು ಬಳಸಿ
        4. ಸೂಕ್ತ ಉದಾಹರಣೆಗಳನ್ನು ನೀಡಿ
        5. ಸಂಭಾಷಣೆಯನ್ನು ಮುಂದುವರೆಸಲು ಸೂಕ್ತ ಪ್ರಶ್ನೆಗಳನ್ನು ಸೇರಿಸಿ

        ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ಸ್ಪಷ್ಟ ಉತ್ತರ ನೀಡಿ.
        Give me response only in Kannada
        """