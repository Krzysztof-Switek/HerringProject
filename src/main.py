from train import Trainer


def main():
    """
    Main entry point for the training pipeline.

    This script initializes and runs the training pipeline based on the settings
    in 'src/config/config.yaml'. You can control the behavior, such as switching
    between 'single' and 'multitask' modes, by modifying the config file.
    """
    print("Initializing the training pipeline...")

    # The Trainer class will automatically load the default config file from 'src/config/config.yaml'
    trainer = Trainer()

    print("Starting the training...")
    trainer.train()

    print("Training pipeline finished.")


if __name__ == "__main__":
    main()
