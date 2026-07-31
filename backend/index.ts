import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import fs from "fs";
import archiver from "archiver";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const server = createServer(app);

  // Serve static files from dist/public in production
  const staticPath =
    process.env.NODE_ENV === "production"
      ? path.resolve(__dirname, "public")
      : path.resolve(__dirname, "..", "dist", "public");

  app.use(express.static(staticPath));

  // API: POST /api/synthesize
  // Accepts multipart form: ref_audio (file, optional), texts (array of strings), output_name (string)
  const upload = multer({ dest: path.resolve(__dirname, "..", "tmp") });

  app.post(
    "/api/synthesize",
    upload.single("ref_audio"),
    express.urlencoded({ extended: true }),
    async (req, res) => {
      try {
        const textsRaw = req.body.texts;
        let texts: string[] = [];
        if (Array.isArray(textsRaw)) {
          texts = textsRaw.filter(Boolean);
        } else if (typeof textsRaw === "string") {
          // single text or newline-separated
          texts = textsRaw.split("\n").map((s) => s.trim()).filter(Boolean);
        }

        if (!texts.length) {
          return res.status(400).json({ error: "No texts provided" });
        }

        const language = typeof req.body.language === 'string' ? req.body.language : undefined;
        const refText = typeof req.body.ref_text === 'string' ? req.body.ref_text : undefined;
        const refAudioPath = req.file ? req.file.path : null;
        const outDir = path.resolve(__dirname, "..", "tmp", `synth_${Date.now()}`);
        fs.mkdirSync(outDir, { recursive: true });

        // Build command to call the Python CLI infer.py for each sentence
        // Use the repo's omnivoice CLI script
        const python = process.env.PYTHON || "python3";
        const modelArg = process.env.OMNIVOICE_MODEL || "k2-fsa/OmniVoice";

        for (let i = 0; i < texts.length; i++) {
          const t = texts[i];
          const outPath = path.join(outDir, `out_${String(i + 1).padStart(3, "0")}.wav`);
          const args = [
            path.resolve(__dirname, "..", "omnivoice", "cli", "infer.py"),
            "--model",
            modelArg,
            "--text",
            t,
            "--output",
            outPath,
          ];
          if (refAudioPath) {
            args.push("--ref_audio", refAudioPath);
          }
          if (refText) {
            args.push("--ref_text", refText);
          }
          if (language) {
            args.push("--language", language);
          }

          // Call synchronously-like by awaiting a child process promise
          await new Promise<void>((resolve, reject) => {
            const p = spawn(python, args, { stdio: "inherit" });
            p.on("close", (code) => {
              if (code === 0) resolve();
              else reject(new Error(`infer.py exit ${code}`));
            });
            p.on("error", reject);
          });
        }

        // Zip the outputs
        const zipName = `synth_${Date.now()}.zip`;
        const zipPath = path.join(outDir, zipName);
        await new Promise<void>((resolve, reject) => {
          const output = fs.createWriteStream(zipPath);
          const archive = archiver("zip", { zlib: { level: 9 } });
          output.on("close", () => resolve());
          archive.on("error", (err) => reject(err));
          archive.pipe(output);
          archive.directory(outDir, false);
          archive.finalize();
        });

        res.download(zipPath, zipName, (err) => {
          // cleanup
          try {
            if (refAudioPath) fs.unlinkSync(refAudioPath);
          } catch {}
        });
      } catch (err: any) {
        console.error(err);
        res.status(500).json({ error: String(err) });
      }
    }
  );

  // Handle client-side routing - serve index.html for all routes
  app.get("*", (_req, res) => {
    res.sendFile(path.join(staticPath, "index.html"));
  });

  const port = process.env.PORT || 3000;

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
