# Self-driving car project:- Lake track.

This project marks my official entry into the field of robotics, while also showcasing my capability in implementing transfer-learning.

## What I got to learn.

    1. Capturing portion of training data that suits best for a model or training literature.
    2. Unfreezing certain layers of a pre-trained model to adapt the model to our particular use-case.
    3. Make custom sampler and collate function to best suit the requirements of the problem.
    4. Always log different iterations of a model (checkpoints) and their respective losses, so we can diagonise what's wrong, incase the loss increases and choose an appropriate starting point and set of hyperparameters for the next training duration. (Very important when working on research-based projects, like this one).
    5. Not being consumed by ego all the time is an important lesson I learnt. It's okay to ask for help, even if it's AI. Vibe-coding the entire thing is what gets you, not the clarifying your concepts.
    6. The importance of normalization: Before this project, I'd never really paid heed to normalization when transforming the input. Since, before this, I was satisfied with getting a >90% accuracy on an image classifier, but robotics is different - Every possible edge case needs to be considered by both the model designer, and the model (during training). Getting >90% accuracy, simply doesn't suffice.

## How it went.

    I started with building a custom dataset class for this project and implementing traning a pre-trained model, by freezing it's CNN (feature-extraction) layers and replacing the fully-connected layer with a custom one after unfreezing it, to train it specifically for our use-case. 

    I had decided not to use the model that had been provided by Nvidia themselves (based on their paper, "End-to-End Deep Learning for Self-Driving Cars"[1]), as part of this project, but I decided to go my way, trying to fit in a model trained on the ImageNet dataset[2], since they are *diverse-enough* to be able to teach model how to extract relevant features from any kind of data.

    **Challenges**:
        1. Realization of the problem being **not exactly** a classification problem, instead was somekind of a 'classifying-regression' one. 
            eg: The steering and throttle not only depended on the frame at that interval, but also on the previous steering and throttle values, as well.

            How I solved it? -> Ignored the regression side of the problem for now, and instead focused on the classification end.
            Result?          -> The model completed the track, but I believe, it did not generalize enough.
            Future scope?    -> Use LSTMs or Transformers to handle the auto-regressive end of the problem, the kind of scenarios, these models are specifically good at.

        2. Bias in training data: The training data was left-biased, i.e. majority of the turns were left turns.
            
            How I solved it? -> Read related papers? Hell nah! I knew the problem was simpler than that. I collected data in the clock-wise direction for the same number of laps as the                 original data.

## Future scope.

    1. Complete working on the hilly track.
    2. Implement LSTMs for the auto-regressive side of the problem.
    3. Implement VLAs to boost generalization and urge Udacity to upgrade this project as a test-bed for people willing to learn and experiment on VLAs for self-driving applications, wherein the car acts on text-prompts from users in natural language, much like what a human cab driver would do.
    4. Release a complete dataset in Kaggle to the further lower the bar of entry for people like me willing to get into the field of self-driving and robotics, without being left to handle inherent discrepancies in the problem and simulator, without having to face it head-on themselves.
    5. Release a complete tutorial on YouTube, but that's been scheduled to later.

## Precautionary notes (to be updated more).
    1. Please install only those versions of the libraries, as specified in the requirements.txt file.
        I had a hard time, getting the simulator working in autonomous mode, only to figure out a version mismatch, and believe me when I say, that it's not fun (I missed Holi, because of it)

## Bibliography:
    [1]Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., ... & Zieba, K. (2016). End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316.
    [2] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3), 211-252.