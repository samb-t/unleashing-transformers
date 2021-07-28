# TODO: move to (sampler?) utils
@torch.no_grad()
def display_output(H, output, vis, ae, model):            

    latents = model.sample() #TODO need to write sample function for EBMS (give option of AIS?)
    q = model.embed(latents)
    images = ae.generator(q.cpu())
    output_win_name = 'samples'
    
    display_images(vis, images, H, win_name=output_win_name)

    return None