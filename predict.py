from entities.AgeCNN import AgeCNN
import torch
from argparse import ArgumentParser
import imageio

def predict_age(model_filename, image_filename):

    # Create instance of the model
    model = AgeCNN()

    # Load instance with latest saved model
    model.load_state_dict(torch.load(model_filename))
    
    # Put model in testing mode since we are making a prediction
    model.eval()

    # Load the image from the input path
    image = imageio.imread(image_filename)

    # Convert image to pytorch tensor object
    new_tensor = torch.tensor(image)

    # Model requires a 4d array (as first one is usually batch size), so unsqueeze adds another of size 1
    new_tensor = new_tensor.unsqueeze(0)

    # Input the image through the model
    output = model(new_tensor)

    # Retrieve in the index of the highest value in the output layer of the model, which is our resulting age
    pred = output.data.max(1, keepdim=True)[1]
    
    # Return only the integer indexed
    return(pred[0][0].item())


# If called on the command line requires args for model and image
# Instead the predict_age function can be called from another module
if __name__ == "__main__":

    # Take user parameters of model directory and image path
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--model-filename', required=True)
    parser.add_argument('--image-filename', required=True)

    # Parse to global args object used in this file
    args = parser.parse_args()

    # Call predict_age with inputs and only print the result if running this file directly
    print(predict_age(args.model_filename, args.image_filename))

