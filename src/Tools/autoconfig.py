def add_wakeword_samples_tool(num_samples: str = "1"):
    """
    Adds 'num_samples' extra samples of the same wakeword.
    """
    try:
        samples = int(num_samples)
        from src.graph import ww_setup, wake_word
        ww_setup.add_wakeword_samples(samples)
        # After adding, reload the templates in the wake_word listening object.
        wake_word._load_templates()
        return f"Successfully added {samples} wakeword samples."
    except Exception as e:
        return f"Error adding wakewords: {e}"

def new_wakeword_tool(**kwargs):
    """
    Creates a new wakeword by deleting the previous ones.
    """
    try:
        from src.graph import ww_setup, wake_word
        ww_setup.new_wakeword_samples()
        # After setting up a new one, reload the listening object.
        wake_word._load_templates()
        return "New wakeword successfully configured."
    except Exception as e:
        return f"Error configuring new wakeword: {e}"

def change_voice_tool(text: str, instruct: str, language: str = "English"):
    """
    Changes the assistant's voice based on a textual description (instruct).
    text: The reference text the new voice will say.
    instruct: Voice description (e.g. 'Female, 24 years old, confident').
    language: The language, defaults to 'English'.
    """
    try:
        from src.graph import qwenVoiceSetup
        if qwenVoiceSetup is None:
            return "QwenVoiceSetup is not enabled."
        qwenVoiceSetup.Change_Voice(text=text, instruct=instruct, language=language)
        return "The voice has been successfully changed and generated."
    except ImportError:
        return "Qwen3 TTS is not enabled. Voice changing tools are unavailable."
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        return f"Error changing the voice: {e}"

def regenerate_voice_tool(**kwargs):
    """
    Regenerates the assistant's voice using previously stored data (in case loading failed or the model changed).
    """
    try:
        from src.graph import qwenVoiceSetup
        if qwenVoiceSetup is None:
            return "QwenVoiceSetup is not enabled."
        success = qwenVoiceSetup.Re_Generate_Voice()
        if success is True:
            return "The voice has been successfully regenerated from the metadata."
        elif success == "CUDA out of memory":
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        else:
            return "Error: Could not regenerate the voice (metadata might not exist)."
    except ImportError:
        return "Qwen3 TTS is not enabled. Voice changing tools are unavailable."
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return "Error: CUDA out of memory. Try enabling FLASH_ATTENTION in config.py or using a smaller model."
        return f"Error regenerating the voice: {e}"
