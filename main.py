import sys
from src.graph import run_aria

def main():
    """
    Main entry point for the ARIA assistant.
    Initializes the processing graph and runs it continuously.
    """
    print("\n--- STARTING ARIA ASSISTANT ---")
    
    try:
        from src.container import Container
        container = Container()
        container.mcp_manager.load_from_config("tools_config.json")
    except Exception as e:
        print(f"\n[Main] ❌ Error initializing container: {e}")
        sys.exit(1)
        
    print("The system will enter standby mode waiting for the activation word.")
    print("Press Ctrl+C to exit.\n")

    try:
        run_aria(container)
    except KeyboardInterrupt:
        print("\n[Main] 👋 Exiting assistant...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main] ❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
