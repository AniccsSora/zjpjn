import torch
import matplotlib.pyplot as plt
import tqdm


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNetEncoder(torch.nn.Module):
    def __init__(self, config):
        # 
        n_ResidualBlock = config['n_ResidualBlock']  # offcial default = 8
        n_levels = config['n_levels']  # offcial default = 4
        input_ch = config.get('input_ch')  # offcial default = 3
        if input_ch is None:
            input_ch = config['input_shape'][2]
        z_dim = config['z_dim']  # offcial default = 10
        bUseMultiResSkips = config['bUseMultiResSkips']  # offcial default = True
        #
        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                        kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(self.max_filters),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(torch.nn.Module):
    def __init__(self, config):
        """
        official default:
            n_ResidualBlock=8,
            n_levels=4,
            z_dim=10,
            output_channels=3,
            bUseMultiResSkips=True
        """
        #
        n_ResidualBlock = config['n_ResidualBlock']
        n_levels = config['n_levels']
        z_dim = config['z_dim']
        output_channels = config['output_channels']
        bUseMultiResSkips = config['bUseMultiResSkips']
        #
        
        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(self.max_filters),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                             kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_1),
                    torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                                 kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(n_filters_1),
                        torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                           kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, z):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        return z


class ResNetAE(torch.nn.Module):
    def __init__(self, config):
        """
        Official default:
             input_shape=(256, 256, 3),
             n_ResidualBlock=8,
             n_levels=4,
             z_dim=128,
             bottleneck_dim=128,
             bUseMultiResSkips=True
        """
        #
        input_shape = config['input_shape']
        n_ResidualBlock = config['n_ResidualBlock']
        n_levels = config['n_levels']
        z_dim = config['z_dim']
        bottleneck_dim = config['bottleneck_dim']
        bUseMultiResSkips = config['bUseMultiResSkips']
        #
        # my logic
        if config.get('__BY_CONDUCTOR__', None) is None:
            raise AssertionError("PLEASE init ResNetAE by ResNetAE_Conductor, and use Config to init.")
        
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)
        
        UseOfficial_default = False
        if UseOfficial_default:
            self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                         input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
            self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                         output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        else:
            self.encoder = ResNetEncoder(config)
            self.decoder = ResNetDecoder(config)

        self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim))

    def decode(self, z):
        h = self.decoder(self.fc2(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.sigmoid(h)

    def forward(self, x):
        return self.decode(self.encode(x))

def ResNetAE_Conductor(config) -> ResNetAE:
    """
    provide ResNetAE, input config will tranfer to E(), D() finally build AE and reutrn. 
    """
    # parser
    input_shape = config['input_shape']  # official default = (256, 256, 3),
    n_ResidualBlock = config['n_ResidualBlock']  # official default = 8,
    n_levels = config['n_levels']  # official default = 4,
    z_dim = config['z_dim']  # official default = 128,
    bottleneck_dim = config['bottleneck_dim']  # official default = 128,
    bUseMultiResSkips = config['bUseMultiResSkips']  # official default = True
    
    # will include E, D params
    config = {
        'input_shape': input_shape,
        'bottleneck_dim': bottleneck_dim,
        'n_ResidualBlock': n_ResidualBlock,
        'n_levels': n_levels,
        'z_dim': z_dim,  # show the same between E(), D()
        'output_channels': input_shape[2],
        'bUseMultiResSkips': True
    }

    # special inner logic
    config.update({'__BY_CONDUCTOR__': "YES"})

    return ResNetAE(config)
    

class ResNetVAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        # Assumes the input to be of shape 256x256
        self.fc21 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc22 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc3 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h1 = self.encoder(x).view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(self.fc3(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VectorQuantizer(torch.nn.Module):
    """
    Implementation of VectorQuantizer Layer from: simplegan.autoencoder.vq_vae
    url: https://simplegan.readthedocs.io/en/latest/_modules/simplegan/autoencoder/vq_vae.html
    """
    def __init__(self, num_embeddings, embedding_dim, commiment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commiment_cost = commiment_cost

        self.embedding = torch.nn.parameter.Parameter(torch.tensor(
            torch.randn(self.embedding_dim, self.num_embeddings)),
            requires_grad=True)

    def forward(self, x):

        flat_x = x.view([-1, self.embedding_dim])

        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_x, self.embedding)
            + torch.sum(self.embedding ** 2, dim=0, keepdim=True)
        )

        encoding_indices = torch.argmax(-distances, dim=1)
        encodings = (torch.eye(self.num_embeddings)[encoding_indices]).to(x.device)
        encoding_indices = torch.reshape(encoding_indices, x.shape[:1] + x.shape[2:])
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, x.shape)

        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = q_latent_loss + self.commiment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        return loss, quantized, perplexity, encoding_indices

    def quantize_encoding(self, x):
        encoding_indices = torch.flatten(x)
        encodings = torch.eye(self.num_embeddings)[encoding_indices]
        quantized = torch.matmul(encodings, torch.transpose(self.embedding, 0, 1))
        quantized = torch.reshape(quantized, torch.Size([-1, self.embedding_dim]) + x.shape[1:])
        return quantized


class ResNetVQVAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(3, 256, 256),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 vq_num_embeddings=512,
                 vq_embedding_dim=64,
                 vq_commiment_cost=0.25,
                 bUseMultiResSkips=True):
        super(ResNetVQVAE, self).__init__()

        assert input_shape[1] == input_shape[2]
        image_channels = input_shape[0]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=self.z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=vq_embedding_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.vq_vae = VectorQuantizer(num_embeddings=vq_num_embeddings,
                                      embedding_dim=vq_embedding_dim,
                                      commiment_cost=vq_commiment_cost)
        self.pre_vq_conv = torch.nn.Conv2d(in_channels=self.z_dim, out_channels=vq_embedding_dim,
                                           kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, perplexity, encodings = self.vq_vae(x)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity, encodings
    

if __name__ == '__main__':
    ae_config = {
    'input_shape': (256, 256, 3),
    'n_ResidualBlock': 8,
    'n_levels': 4,
    'z_dim': 10,  
    'bottleneck_dim': 128,  # 真正的 latent vector size
    'bUseMultiResSkips': True
    }
    decoder_config = {
        'n_ResidualBlock': 8,
        'n_levels': 4,
        'z_dim': 10,  # the same between E(), D()
        'output_channels': 3,
        'bUseMultiResSkips': True
    }
    encoder_config = {
        'n_ResidualBlock': 8,
        'n_levels': 4,
        'input_ch': 3,
        'z_dim': 10,  # the same between E(), D()
        'bUseMultiResSkips': True
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ResNetEncoder(config=encoder_config).to(device)
    
    decoder = ResNetDecoder(config=decoder_config).to(device)
    
    ae = ResNetAE_Conductor(config=ae_config).to(device)

    #
    test_input_E = torch.rand(10, 3, ae_config['input_shape'][0], ae_config['input_shape'][1]).to(device)
    out = encoder(test_input_E)

    test_input_O = torch.rand(10, 10, 16, 16).to(device)
    out = decoder(test_input_O)
    #
    encoder_num_layers = sum(1 for _ in encoder.named_modules())
    print("Number of layers (encoder):", encoder_num_layers)
    print("input.shape =", test_input_E.shape)
    print("output.shape =", encoder(test_input_E).shape)
    #
    decoder_num_layers = sum(1 for _ in decoder.named_modules())
    print("Number of layers (decoder):", decoder_num_layers)
    print("input.shape =", test_input_O.shape)
    print("output.shape =", decoder(test_input_O).shape)
    #
    ae_num_layers = sum(1 for _ in ae.named_modules())
    print("Number of layers (AutoEncoder):", ae_num_layers)
    print("input.shape =", test_input_E.shape)
    print("output.shape =", ae(test_input_E).shape)

    # ======================================
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(),
                                 lr=1e-1,
                                 weight_decay=1e-8)
    epochs = 10
    losses = []

    pbar = tqdm.tqdm(range(epochs))

    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        for image in test_input_E:
            optimizer.zero_grad()

            # A image to batch
            ground_batch = image.unsqueeze(0)

            # Output of Autoencoder
            reconstructed = ae(ground_batch.to(device))

            # Calculating the loss function
            loss = loss_function(reconstructed, ground_batch)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update

            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.item())


    # Plotting the last 100 values
    plt.plot(losses)
    plt.show()
