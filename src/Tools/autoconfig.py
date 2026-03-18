def add_wakeword_samples_tool(num_samples: str = "1", container=None):
    """
    Adds 'num_samples' extra samples of the same wakeword.
    """
    try:
        if container is None:
            return "Error: Container not provided."
        samples = int(num_samples)
        container.ww_setup.add_wakeword_samples(samples)
        # After adding, reload the templates in the wake_word listening object.
        container.wake_word._load_templates()
        return f"Successfully added {samples} wakeword samples."
    except Exception as e:
        return f"Error adding wakewords: {e}"

def new_wakeword_tool(container=None, **kwargs):
    """
    Creates a new wakeword by deleting the previous ones.
    """
    try:
        if container is None:
            return "Error: Container not provided."
        container.ww_setup.new_wakeword_samples()
        # After setting up a new one, reload the listening object.
        container.wake_word._load_templates()
        return "New wakeword successfully configured."
    except Exception as e:
        return f"Error configuring new wakeword: {e}"

def change_voice_tool(text: str, instruct: str, language: str = "English", container=None):
    """
    Changes the assistant's voice based on a textual description (instruct).
    text: The reference text the new voice will say.
    instruct: Voice description (e.g. 'Female, 24 years old, confident').
    language: The language, defaults to 'English'.
    """
    try:
        if container is None:
            return "Error: Container not provided."
        if not hasattr(container, 'qwen_voice_setup') or container.qwen_voice_setup is None:
            return "QwenVoiceSetup is not enabled. Voice changing tools are unavailable."
        container.qwen_voice_setup.Change_Voice(text=text, instruct=instruct, language=language)
        return "The voice has been successfully changed and generated."
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        return f"Error changing the voice: {e}"

def regenerate_voice_tool(container=None, **kwargs):
    """
    Regenerates the assistant's voice using previously stored data (in case loading failed or the model changed).
    """
    try:
        if container is None:
            return "Error: Container not provided."
        if not hasattr(container, 'qwen_voice_setup') or container.qwen_voice_setup is None:
            return "QwenVoiceSetup is not enabled. Voice changing tools are unavailable."
        success = container.qwen_voice_setup.Re_Generate_Voice()
        if success is True:
            return "The voice has been successfully regenerated from the metadata."
        elif success == "CUDA out of memory":
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        else:
            return "Error: Could not regenerate the voice (metadata might not exist)."
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        return f"Error regenerating the voice: {e}"
