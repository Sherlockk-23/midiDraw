from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    model_name="pann",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

fad_score = frechet.score(
    "/Users/integer/Downloads/waves_gt", 
    "/Users/integer/Downloads/waves_img", 
    dtype="float32"
)