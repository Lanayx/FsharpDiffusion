#r "nuget:TorchSharp-cuda-windows, 0.100.3"
#r "nuget: TorchVision, 0.100.3"
#r "nuget: PureHDF, 1.0.0-alpha.25"
#r "nuget: SkiaSharp, 2.88.3"

open System.Collections.Generic
open TorchSharp
open System
open PureHDF
open SkiaSharp

open type TorchSharp.torch
open type TorchSharp.torchvision
open type TorchSharp.torchvision.io
open type TorchSharp.torch.utils.data
open System.IO

type ResidualConvBlock(in_channels: int64, out_channels: int64, is_res: bool, t_device) as this =
    inherit nn.Module<Tensor,Tensor>("ResidualConvBlock")  
   

    // Check if input and output channels are the same for the residual connection
    let same_channels = in_channels = out_channels

    // First convolutional layer
    let conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3L, stride = 1L, padding = 1L, device = t_device),   // 3x3 kernel with stride 1 and padding 1
        nn.BatchNorm2d(out_channels, device = t_device),   // Batch normalization
        nn.GELU()  // GELU activation function
    )

    // Second convolutional layer
    let conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, device = t_device),   // 3x3 kernel with stride 1 and padding 1
        nn.BatchNorm2d(out_channels, device = t_device),   // Batch normalization
        nn.GELU()  // GELU activation function
    )
    do this.RegisterComponents()

    override _.forward(x) =
        // If using residual connection
        if is_res then
            // Apply first convolutional layer
            let x1 = conv1.forward(x)

            // Apply second convolutional layer
            let x2 = conv2.forward(x1)

            // If input and output channels are the same, add residual connection directly
            let out =
                if same_channels then
                    x + x2
                else
                    // If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                    let shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernelSize = 1L, stride=1L, padding=0L, device = t_device)
                    shortcut.forward(x) + x2
            // print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")
         
            // Normalize output tensor
            out / 1.414.ToTensor(t_device)

        // If not using residual connection, return output of second convolutional layer
        else
            let x1 = conv1.forward(x)
            let x2 = conv2.forward(x1)
            x2

type UnetUp(in_channels: int64, out_channels: int64, t_device) as this =
    inherit nn.Module<IList<Tensor>,Tensor>("UnetUp") 
        
    // Create a list of layers for the upsampling block
    // The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
    let layers = [|
        nn.ConvTranspose2d(in_channels, out_channels, 2, 2, device = t_device) :> nn.Module<_,_>
        new ResidualConvBlock(out_channels, out_channels, false, t_device)
        new ResidualConvBlock(out_channels, out_channels, false, t_device)
    |]
    
    // Use the layers to create a sequential model
    let model = nn.Sequential(layers)

    do this.RegisterComponents()
    override _.forward(tensors) =       
        // Concatenate the input tensor x with the skip connection tensor along the channel dimension
        let x = cat(tensors, 1)
        
        // Pass the concatenated tensor through the sequential model and return the output
        model.forward(x)
        
type UnetDown(in_channels: int64, out_channels: int64, t_device) as this =
    inherit nn.Module<Tensor,Tensor>("UnetDown") 
        
    // Create a list of layers for the downsampling block
    // Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
    let layers = [|
        new ResidualConvBlock(in_channels, out_channels, false, t_device) :> nn.Module<_,_>
        new ResidualConvBlock(out_channels, out_channels, false, t_device)
        nn.MaxPool2d(2)
    |]
    
    // Use the layers to create a sequential model
    let model = nn.Sequential(layers)

    do this.RegisterComponents()
    override _.forward(x) =
        // Pass the input through the sequential model and return the output
        model.forward(x)

type EmbedFC(input_dim: int64, emb_dim: int64, t_device) as this =
    inherit nn.Module<Tensor,Tensor>("EmbedFC") 
    (*
    This class defines a generic one layer feed-forward neural network for embedding input data of
    dimensionality input_dim to an embedding space of dimensionality emb_dim.
    *)
    let input_dim = input_dim
    
    // define the layers for the network
    let layers = [|
        nn.Linear(input_dim, emb_dim, device = t_device) :> nn.Module<_,_>
        nn.GELU()
        nn.Linear(emb_dim, emb_dim, device = t_device)
    |]
    
    // create a PyTorch sequential model consisting of the defined layers
    let model = nn.Sequential(layers)

    do this.RegisterComponents()
    override _.forward(x) =
        // flatten the input tensor
        let x = x.view(-1, input_dim)
        // apply the model layers to the flattened tensor
        model.forward(x)

let unorm(x: Tensor) =
    // unity norm. results in range of [0,1]
    // assume x (h,w,3)
    let xmax = x.max()
    let xmin = x.min()
    (x - xmin)/(xmax - xmin)

let norm_all(store: Tensor, n_t, n_s) =
    // runs unity norm on all timesteps of all samples
    let nstore = zeros_like(store)
    for t in 1L..n_t do
        for s in 1L..n_s do
            nstore[t-1L,s-1L] <- unorm(store[t-1L,s-1L])
    nstore

let drawImage (image: Tensor) =
    let byteImage = (image * 255.0.ToScalar()).``to``(ScalarType.Byte).contiguous()
    let stream = new MemoryStream()
    torchvision.io.write_png(byteImage, stream, SkiaImager())
    stream.Seek(0L, SeekOrigin.Begin) |> ignore        
    let imageInfo = SKImageInfo(16, 16, SKColorType.Bgra8888, SKAlphaType.Unpremul)
    let bm = SKBitmap.Decode(stream, imageInfo)   
    bm

let plot_sample(x_gen_store: Tensor) =    
    let n_timestamps = x_gen_store.shape[0]
    let n_samples = x_gen_store.shape[1]

    let nsx_gen_store = norm_all(
        x_gen_store, n_timestamps, n_samples)   // unity norm to put in range [0,1] for np.imshow

    let images: SKBitmap array = [|
        for sample in 0L..(n_samples-1L) do
            let lastTimeStampData = nsx_gen_store[n_timestamps-1L]   
            let lastImage = lastTimeStampData[sample-1L]
            yield drawImage lastImage
    |]

    images

type CustomDataset(sfilename, lfilename, transform: ITransform, null_context, t_device) =
    inherit Dataset()

    let spritesDataSet = H5File.OpenRead(sfilename).Dataset("default")
    let slabelsDataSet = H5File.OpenRead(lfilename).Dataset("default")
    do Console.WriteLine($"sprite shape: %A{spritesDataSet.Space.Dimensions}")
    do Console.WriteLine($"labels shape: %A{slabelsDataSet.Space.Dimensions}")
    let spritesData = spritesDataSet.Read()
    let spritesTensor = 
        torch.from_array(spritesData, t_device).view(89400, 16, 16, 3)
    let labelsData = slabelsDataSet.Read<float>()
    let labelsTensor = 
        torch.from_array(labelsData, t_device).view(89400, 5)
                
    // Return the number of images in the dataset
    override _.get_Count() =
        spritesDataSet.Space.Dimensions[0] |> Operators.int64
    
    // Get the image and label at a given index
    override _.GetTensor(index) =
        // Return the image and label as a tuple
        let d = Dictionary()
        let x = spritesTensor[index]
        let image = transform.call(x)
        d.Add("data", image)
        let label: torch.Tensor =
            if null_context then
                0L.ToTensor(t_device).``to``(torch.int64)
            else                
                labelsTensor[index].``to``(torch.int64)
        d.Add("label", label)
        d

module Utilities =
    let transform t_device = 

        let to_tensor = 
            {
                new ITransform with 
                    member _.call(img) =
                        img
                            // put it from HWC to CHW format
                            .permute(2, 0, 1)
                            .contiguous()
                            .``to``(ScalarType.Float32)
                            .div(255f.ToScalar())
                            // hack for torchvision
                            .reshape(1, 3, 16, 16)
            }
        // remove hack for torchvision
        let removeHack = 
            {
                new ITransform with 
                    member _.call(img) =
                        img[0]
            }

        transforms.Compose([|
            to_tensor                                                                      // from [0,255] to range [0.0,1.0]
            transforms.Normalize([|0.5; 0.5; 0.5|], [|0.5; 0.5; 0.5|], device = t_device)  // range [-1,1]
            removeHack
        |])