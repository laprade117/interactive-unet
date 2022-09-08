import sys
from unet_app import app

if __name__ == "__main__":
    args = sys.argv[1:]

    port = 34598

    if args[0] == '-p':
        port = int(args[1])      
    
    app.run(port=port)