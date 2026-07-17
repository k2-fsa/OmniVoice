# Tips & Notes

- **Combination of `ref_audio` and `instruct`**:
  When both `ref_audio` and `instruct` are provided and they **conflict**, the model will most likely follow the style of the reference audio. When the two are **consistent**, `instruct` can improve cloning stability for the attributes it describes. A typical example is **Chinese dialect cloning**: provide both dialect reference audio and a matching dialect instruct (e.g., `ref_audio="sichuan.wav", instruct="四川话"`) for more stable dialect output.

- **Short Audio Generation**:
  The model may not reliably generate short audio clips (e.g., 1–2 seconds) without reference audio. If you need to generate short clips, provide reference audio to the model.

- **Min Nan Chinese (Hokkien) Input Format**:
  Min Nan Chinese (闽南语, also known as Hokkien) can only be synthesized using [Tai-lo romanization](https://en.wikipedia.org/wiki/T%C3%A2i-l%C3%B4) as input; Chinese characters are not supported for Min Nan Chinese in the current model version.
