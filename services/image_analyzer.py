import os
import time
import base64
import io
import traceback
import requests
from pathlib import Path
from typing import Union
from PIL import Image
from openai import AsyncOpenAI
import asyncio
from typing import Optional


class ImageAnalyzer:
    def __init__(self, hf_token: str = None):
        """Initialize the ImageAnalyzer with robust error handling and validation."""
        # Store the token (either provided or from environment)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be provided either as argument or in environment")
            
        # Client will be initialized asynchronously
        self.client: Optional[AsyncOpenAI] = None
        
        # Configuration parameters
        self.max_retries = 3
        self.timeout = 30
        self.retry_delay = 1  # seconds

    def _setup_logging_header(self, message: str):
        """Helper for consistent debug logging headers."""
        print(f"\n[DEBUG] {'=' * 40}")
        print(f"[DEBUG] {message}")
        print(f"[DEBUG] {'-' * 40}")

    def _setup_logging_footer(self, message: str):
        """Helper for consistent debug logging footers."""
        print(f"[DEBUG] {'-' * 40}")
        print(f"[DEBUG] {message}")
        print(f"[DEBUG] {'=' * 40}\n")

    def _validate_environment(self):
        """Validate required environment variables and dependencies."""
        try:
            if not os.getenv("HF_TOKEN"):
                raise EnvironmentError("HF_TOKEN environment variable is not set")
            
            print(f"[DEBUG] HF_TOKEN exists: {'HF_TOKEN' in os.environ}")
            print(f"[DEBUG] HF_TOKEN length: {len(os.getenv('HF_TOKEN', ''))}")
            
            # Test PIL availability
            Image.new('RGB', (1, 1))
            
        except Exception as e:
            self._log_critical_error("Environment validation failed", e)
            raise

    async def initialize(self):
        """Asynchronously initialize and validate the OpenAI client."""
        self._setup_logging_header("Initializing ImageAnalyzer")
        self.client = await self._initialize_client()
        self._setup_logging_footer("Analyzer ready")

    async def _initialize_client(self) -> AsyncOpenAI:
        """Initialize and validate the OpenAI client with retries."""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"[DEBUG] Client initialization attempt {attempt + 1}/{max_attempts}")
                
                client = AsyncOpenAI(
                    base_url="https://router.huggingface.co/v1",
                    api_key=self.hf_token  # Use the stored token
                )
                
                # Immediate test of the client
                test_response = await client.chat.completions.create(
                    model="zai-org/GLM-4.1V-9B-Thinking",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                
                if not test_response.choices:
                    raise ValueError("Empty test response from API")
                
                print("[DEBUG] Client test successful")
                return client
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    self._log_critical_error("Client initialization failed after retries", e)
                    raise
                
                print(f"[WARNING] Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(self.retry_delay)

    def _log_critical_error(self, context: str, error: Exception):
        """Standardized error logging for critical failures."""
        print(f"\n[CRITICAL] {context}")
        print(f"[DEBUG] Error type: {type(error).__name__}")
        print(f"[DEBUG] Error message: {str(error)}")
        print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        if hasattr(error, 'response'):
            print(f"[DEBUG] Response status: {getattr(error.response, 'status_code', 'N/A')}")
            print(f"[DEBUG] Response text: {getattr(error.response, 'text', 'N/A')[:200]}...")

    async def analyze(self, image_input: Union[str, Image.Image, bytes], prompt: str = None, vehicle_info: dict = None) -> str:
        """
        Robust image analysis with comprehensive error handling and retries.
        
        Args:
            image_input: Image in various formats (URL, path, PIL Image, bytes)
            prompt: Optional custom prompt to augment the analysis
            vehicle_info: Optional dict with vehicle context (year, brand, model)
            
        Returns:
            Analysis result as string or error message if analysis fails
        """
        if not self.client:
            raise RuntimeError("ImageAnalyzer not initialized. Call await .initialize() first.")

        self._setup_logging_header("Starting image analysis")
        print(f"[DEBUG] Input type: {type(image_input)}")
        
        try:
            # Build the base prompt with optional vehicle context
            base_prompt = self._build_base_prompt(prompt, vehicle_info)
            
            # Prepare the image for analysis
            image_url = self._prepare_image(image_input)
            
            # Perform the analysis with retries
            return await self._perform_analysis_with_retries(base_prompt, image_url)
            
        except Exception as e:
            error_msg = self._handle_analysis_error(e)
            return error_msg

    def _build_base_prompt(self, custom_prompt: str = None, vehicle_info: dict = None) -> str:
        """Construct the analysis prompt with optional vehicle context."""
        base_prompt = """
        Analyze this vehicle image in detail and provide:
        1. Make, model, and year estimation
        2. Visible features and condition
        3. Any damage or issues
        4. Relevant maintenance considerations
        """

        if vehicle_info:
            print(f"[DEBUG] Vehicle info: {vehicle_info}")
            
            # Validate vehicle info structure
            required_keys = {'year', 'brand', 'model'}
            if not all(key in vehicle_info for key in required_keys):
                print("[WARNING] Incomplete vehicle info provided")
            else:
                base_prompt += (
                    f"\nContext: This is a {vehicle_info.get('year', 'unknown year')} "
                    f"{vehicle_info.get('brand', 'unknown brand')} {vehicle_info.get('model', 'unknown model')}. "
                    f"Focus on issues common for this model."
                )
                
                # Add verification check
                expected_model = f"{vehicle_info.get('year')} {vehicle_info.get('brand')} {vehicle_info.get('model')}"
                base_prompt = f"Verify if this image matches a {expected_model}. If not, note the differences.\n{base_prompt}"

        if custom_prompt:
            print(f"[DEBUG] Custom prompt: {custom_prompt}")
            base_prompt = f"{custom_prompt}\n{base_prompt}"

        return base_prompt

    def _prepare_image(self, image_input: Union[str, Image.Image, bytes]) -> str:
        """
        Convert various image formats to base64 URL with comprehensive validation.
        
        Args:
            image_input: Image in various formats (URL, path, PIL Image, bytes)
            
        Returns:
            Image as base64 data URL or original URL if already web-accessible
            
        Raises:
            ValueError: If image input is invalid or processing fails
        """
        print("\n[DEBUG] Preparing image input")
        print(f"[DEBUG] Input type: {type(image_input)}")
        
        try:
            if isinstance(image_input, str):
                return self._process_string_input(image_input)
            elif isinstance(image_input, Image.Image):
                return self._process_pil_image(image_input)
            elif isinstance(image_input, bytes):
                return self._process_bytes_input(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
                
        except Exception as e:
            self._log_critical_error("Image preparation failed", e)
            raise ValueError(f"Image preparation error: {str(e)}")

    def _process_string_input(self, image_input: str) -> str:
        """Process string input which could be URL, base64, or file path."""
        if image_input.startswith("http"):
            print("[DEBUG] HTTP URL detected")
            if "cloudinary.com" in image_input:
                # Cloudinary URL - we can use it directly
                print("[DEBUG] Cloudinary URL detected")
                return image_input
            return self._process_url_input(image_input)
        elif image_input.startswith("data:image"):
            print("[DEBUG] Base64 image data detected")
            return image_input
        else:
            print("[DEBUG] Local file path suspected")
            return self._process_file_path_input(image_input)
        

    def _process_url_input(self, url: str) -> str:
        """Process URL input with validation and fallback to download."""
        try:
            # First try HEAD request for efficiency
            response = requests.head(url, timeout=5, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image (Content-Type: {content_type})")
                
            print("[DEBUG] URL is accessible and points to an image")
            return url
            
        except Exception as head_error:
            print(f"[DEBUG] HEAD request failed, falling back to download: {str(head_error)}")
            try:
                response = requests.get(url, timeout=10, stream=True)
                response.raise_for_status()
                
                # Verify content type in case HEAD failed
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"Downloaded content is not an image (Content-Type: {content_type})")
                    
                return f"data:image/jpeg;base64,{base64.b64encode(response.content).decode()}"
                
            except Exception as download_error:
                raise ValueError(f"Failed to process URL: {str(download_error)}") from download_error

    def _process_file_path_input(self, file_path: str) -> str:
        """Process local file path input."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
                
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
                
            # Check file size (limit to 10MB for example)
            max_size = 10 * 1024 * 1024  # 10MB
            if path.stat().st_size > max_size:
                raise ValueError(f"File too large (>{max_size/1024/1024}MB)")
                
            with open(path, "rb") as f:
                data = f.read()
                print(f"[DEBUG] File size: {len(data)} bytes")
                return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
                
        except Exception as e:
            raise ValueError(f"File processing error: {str(e)}") from e

    def _process_pil_image(self, image: Image.Image) -> str:
        """Process PIL Image input."""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=90)
            data = buffered.getvalue()
            print(f"[DEBUG] Image size: {len(data)} bytes")
            return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        except Exception as e:
            raise ValueError(f"PIL Image processing error: {str(e)}") from e

    def _process_bytes_input(self, image_data: bytes) -> str:
        """Process raw bytes input."""
        try:
            # Basic validation - check if it looks like an image
            if len(image_data) < 4:  # Minimum size for even tiny images
                raise ValueError("Image data too small to be valid")
                
            print(f"[DEBUG] Data size: {len(image_data)} bytes")
            return f"data:image/jpeg;base64,{base64.b64encode(image_data).decode()}"
        except Exception as e:
            raise ValueError(f"Bytes processing error: {str(e)}") from e

    async def _perform_analysis_with_retries(self, prompt: str, image_url: str) -> str:
        """Perform the actual analysis with retry logic."""
        if not self.client:
            raise RuntimeError("ImageAnalyzer not initialized.")
        for attempt in range(self.max_retries):
            try:
                print(f"\n[DEBUG] Analysis attempt {attempt + 1}/{self.max_retries}")
                print(f"[DEBUG] Prompt: {prompt[:100]}...")
                print(f"[DEBUG] Image URL: {image_url[:100]}...")
                
                start_time = time.time()
                response = await self.client.chat.completions.create(
                    model="zai-org/GLM-4.1V-9B-Thinking",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }],
                    max_tokens=500,
                    timeout=self.timeout
                )
                elapsed = time.time() - start_time
                
                print(f"[DEBUG] Analysis completed in {elapsed:.2f}s")
                return self._process_response(response)
                
            except Exception as e:
                self._handle_attempt_error(attempt, e)
                
                if attempt == self.max_retries - 1:
                    raise  # Re-raise the last error after all retries
                
                await asyncio.sleep(self.retry_delay)

    def _process_response(self, response) -> str:
        """Process and validate the API response."""
        if not hasattr(response, 'choices') or not response.choices:
            print("[ERROR] Unexpected response format - no choices")
            print(f"[DEBUG] Full response: {response}")
            raise ValueError("Unexpected response format from model")
            
        content = response.choices[0].message.content
        print(f"[DEBUG] Response content: {content[:200]}...")
        
        # Post-processing checks
        if "I cannot analyze" in content or "I can't see" in content:
            print("[WARNING] Model indicates it can't analyze the image")
            return "The model couldn't analyze this image. Please try with a clearer image."
            
        return content

    def _handle_attempt_error(self, attempt: int, error: Exception):
        """Handle and log errors during analysis attempts."""
        print(f"\n[ERROR] Attempt {attempt + 1} failed")
        print(f"[DEBUG] Error type: {type(error).__name__}")
        print(f"[DEBUG] Error message: {str(error)}")
        
        if hasattr(error, 'response'):
            print(f"[DEBUG] Response status: {getattr(error.response, 'status_code', 'N/A')}")
            print(f"[DEBUG] Response text: {getattr(error.response, 'text', 'N/A')[:200]}...")

    def _handle_analysis_error(self, error: Exception) -> str:
        """Generate appropriate user-facing error messages."""
        self._log_critical_error("Analysis failed completely", error)
        
        error_str = str(error).lower()
        
        if "image" in error_str or "vision" in error_str:
            return "I can see an image but having trouble analyzing it. Please describe what you see."
        elif "timeout" in error_str:
            return "The analysis took too long and timed out. Please try again with a different image."
        elif "invalid image" in error_str or "unsupported" in error_str:
            return "The provided image format is not supported. Please try with a JPEG or PNG image."
        else:
            return f"Analysis error: {str(error)}"
    

    