"""
Módulo de Text-To-Speech (TTS) usando edge-tts o KokoroTTS + RVC opcional.

Flujo general:
1. Texto -> (edge-tts | KokoroTTS) -> bytes de audio (MP3 o WAV)
2. (Opcional) WAV -> RVC -> WAV convertido
3. Decodificar y reproducir con sounddevice / soundfile
"""

import asyncio
import base64
import io
import shutil
import subprocess
import sys
from pathlib import Path
import threading

import edge_tts
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from config import (
    VOICE_NAME,
    RVC_API_URL,
    RVC_MODEL,
    USE_RVC,
    USE_KOKORO,
    KOKORO_LANG,
    KOKORO_VOICE,
    KOKORO_SPEED,
)


class TTSEngine:
    """
    Motor TTS que puede usar:
      - edge-tts (por defecto)
      - KokoroTTS (si USE_KOKORO == True y está instalado)
    y, opcionalmente, RVC para transformar la voz.

    - Si USE_KOKORO == True y Kokoro funciona -> KokoroTTS
    - Si USE_KOKORO == True pero falla Kokoro -> fallback a edge-tts
    - Si USE_KOKORO == False -> edge-tts

    Además:
    - Si USE_RVC == False  -> sin RVC
    - Si USE_RVC == True   -> (edge-tts | Kokoro) + RVC (con fallback al audio original si falla)
    """

    def __init__(self):
        """
        Inicializa el motor TTS.

        - Configura:
            * self.voice (edge-tts)
            * self.use_kokoro y pipeline de Kokoro (si está instalado)
            * self.use_rvc y carga el modelo RVC en el servidor
        """
        # ----- Configuración edge-tts -----
        self.voice = VOICE_NAME

        # ----- Configuración Kokoro -----
        self.use_kokoro = bool(USE_KOKORO)
        self.kokoro_lang = KOKORO_LANG
        self.kokoro_voice = KOKORO_VOICE
        self.kokoro_speed = float(KOKORO_SPEED)
        self._kokoro_available = False
        self._kokoro_pipeline = None

        if self.use_kokoro:
            try:
                from kokoro import KPipeline  # import lazy para no romper si no está instalado

                self._kokoro_pipeline = KPipeline(lang_code=self.kokoro_lang)
                self._kokoro_available = True
                print(
                    f"[Kokoro] Inicializado correctamente. "
                    f"lang={self.kokoro_lang}, voice={self.kokoro_voice}"
                )
            except Exception as e:
                self._kokoro_available = False
                self.use_kokoro = False
                print(
                    "[Kokoro] No se pudo inicializar KokoroTTS. "
                    "Se usará edge-tts. Detalle:",
                    e,
                    file=sys.stderr,
                )

        # ----- Configuración RVC -----
        self.use_rvc = bool(USE_RVC)
        self._rvc_available = False

        if self.use_rvc:
            try:
                # Ajusta el endpoint / payload si tu API RVC usa otros nombres.
                resp = requests.post(
                    f"{RVC_API_URL}/models/{RVC_MODEL}",
                    timeout=60,
                )
                resp.raise_for_status()
                self._rvc_available = True
                print(f"[RVC] Modelo cargado correctamente: {RVC_MODEL}")
            except Exception as e:
                # Si falla la carga del modelo, deshabilitamos RVC y seguimos sin transformar.
                self._rvc_available = False
                self.use_rvc = False
                print(
                    f"[RVC] No se pudo cargar el modelo '{RVC_MODEL}'. "
                    f"Se usará la voz sin RVC. Detalle: {e}",
                    file=sys.stderr,
                )

    # ───────────────────────────────────────────────
    # Helpers internos: RVC
    # ───────────────────────────────────────────────
    def _maybe_apply_rvc(self, audio_bytes: bytes) -> bytes:
        """
        Si RVC está habilitado y cargado, convierte:
            audio_bytes (MP3/WAV/lo que sea)
        -> WAV -> RVC -> WAV convertido.

        Si ocurre cualquier error, devuelve el audio original (fallback).
        """
        if not (self.use_rvc and self._rvc_available):
            return audio_bytes

        try:
            # 1) Audio -> WAV en memoria (por si no lo está ya)
            wav_bytes = self._convert_to_wav_bytes(audio_bytes)
            # 2) WAV -> RVC -> WAV convertido
            converted_wav = self.send_wav_bytes_to_rvc(wav_bytes)
            return converted_wav
        except Exception as e:
            # Fallback: usar audio original
            print(
                f"[RVC] Error al aplicar RVC, usando audio original. "
                f"Detalle: {e}",
                file=sys.stderr,
            )
            return audio_bytes

    @staticmethod
    def _convert_to_wav_bytes(audio_bytes: bytes) -> bytes:
        """
        Intentar convertir bytes de audio (por ejemplo MP3) a WAV bytes en memoria.

        1) Intentar decodificar con soundfile (libsndfile) y reescribir WAV.
        2) Si falla, usar ffmpeg (si está disponible) para convertir in-memory.
        """
        try:
            bio = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(bio, dtype="int16")

            out = io.BytesIO()
            sf.write(out, data, samplerate, format="WAV")
            return out.getvalue()
        except Exception:
            # Intentar ffmpeg
            if shutil.which("ffmpeg"):
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        "pipe:0",
                        "-f",
                        "wav",
                        "pipe:1",
                    ],
                    input=audio_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if proc.returncode != 0:
                    raise RuntimeError(
                        "ffmpeg falló al convertir audio a WAV: "
                        + proc.stderr.decode(errors="ignore")
                    )

                return proc.stdout

            raise RuntimeError(
                "No se pudo convertir audio a WAV en memoria. Instala ffmpeg "
                "o habilita soporte MP3 en libsndfile."
            )

    @staticmethod
    def send_wav_bytes_to_rvc(wav_bytes: bytes) -> bytes:
        """
        Codifica los bytes WAV en base64 y los envía a la API RVC en JSON.
        Devuelve los bytes de audio WAV convertidos.

        IMPORTANTE:
        - Si la API responde con 4xx/5xx se lanza una excepción
          (que será capturada arriba para hacer fallback).
        """
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        resp = requests.post(
            f"{RVC_API_URL}/convert",
            json={"audio_data": audio_b64},
            headers=headers,
            timeout=30,
        )

        if resp.status_code != 200:
            print("[error] RVC response status:", resp.status_code)
            try:
                print("[error] RVC response json:", resp.json())
            except Exception:
                print("[error] RVC response text:", resp.text)
            resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "audio" in content_type or resp.content:
            print(
                "Respuesta de RVC: recibidos bytes de audio. Longitud:",
                len(resp.content),
            )
            return resp.content

        try:
            print("Respuesta de RVC (json):", resp.json())
        except Exception:
            print("Respuesta de RVC (status):", resp.status_code)
        raise RuntimeError("La API RVC no devolvió audio.")

    # ───────────────────────────────────────────────
    # Helpers internos: generación con edge-tts vs Kokoro
    # ───────────────────────────────────────────────
    async def _synthesize_edge_bytes_async(self, text: str) -> bytes:
        """
        Genera audio desde texto usando edge-tts y devuelve los bytes producidos
        (normalmente MP3 u otro formato comprimido).
        """
        communicate = edge_tts.Communicate(text, voice=self.voice, rate="+0%")

        audio_buf = bytearray()
        async for msg in communicate.stream():
            if msg.get("type") == "audio":
                data = msg.get("data") or msg.get("audio")
                if data is None:
                    continue

                if isinstance(data, str):
                    try:
                        chunk = base64.b64decode(data)
                    except Exception:
                        continue
                else:
                    chunk = data

                audio_buf.extend(chunk)

        return bytes(audio_buf)

    async def _synthesize_kokoro_bytes_async(self, text: str) -> bytes:
        """
        Genera audio desde texto usando KokoroTTS y devuelve WAV bytes.

        - Usa KPipeline(lang_code=KOKORO_LANG)
        - Usa la voz KOKORO_VOICE
        - Concatena todos los segmentos de audio.
        - Aplica un ajuste sencillo de velocidad basado en KOKORO_SPEED
          (time-stretch naive por resampling).
        """
        if not (self.use_kokoro and self._kokoro_available and self._kokoro_pipeline):
            # Seguridad extra: si Kokoro no está disponible, devolvemos vacío
            # y el caller se encargará de fallback.
            return b""

        # Kokoro genera (gs, ps, audio) donde audio es un np.ndarray (mono) a 24 kHz
        generator = self._kokoro_pipeline(text, voice=self.kokoro_voice)

        chunks = []
        samplerate = 24000
        for _, _, audio in generator:
            # audio es un np.ndarray
            chunks.append(audio)

        if not chunks:
            return b""

        audio = np.concatenate(chunks)

        # Ajuste sencillo de velocidad (no cambia tono de forma independiente)
        if self.kokoro_speed != 1.0 and self.kokoro_speed > 0:
            orig_len = len(audio)
            new_len = int(orig_len / self.kokoro_speed)
            if new_len > 0:
                x = np.linspace(0.0, 1.0, orig_len, endpoint=False)
                x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                audio = np.interp(x_new, x, audio).astype(audio.dtype)

        # Escribimos a WAV en memoria
        buf = io.BytesIO()
        sf.write(buf, audio, samplerate, format="WAV")
        return buf.getvalue()

    # ───────────────────────────────────────────────
    # API pública async / sync
    # ───────────────────────────────────────────────
    async def speak_async(self, text: str):
        """
        Genera audio desde texto y lo reproduce (versión async).

        Flujo:
        1. (KokoroTTS | edge-tts) -> audio_bytes
        2. (Opcional) audio_bytes -> RVC -> audio_bytes_modificados
        3. Decodificar y reproducir
        """
        if not text:
            return

        # 1) Elegir motor TTS
        if self.use_kokoro and self._kokoro_available:
            audio_bytes = await self._synthesize_kokoro_bytes_async(text)
            if not audio_bytes:
                # Si por alguna razón Kokoro no generó nada, fallback a edge-tts
                audio_bytes = await self._synthesize_edge_bytes_async(text)
        else:
            audio_bytes = await self._synthesize_edge_bytes_async(text)

        if not audio_bytes:
            return

        # 1.5) Aplicar RVC si procede (con fallback interno)
        audio_bytes = self._maybe_apply_rvc(audio_bytes)

        # 2) Decodificar audio en memoria y reproducir
        bio = io.BytesIO(audio_bytes)

        try:
            data, samplerate = sf.read(bio, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
            print(f"🔊 TTS reproducido en memoria (samplerate={samplerate})")
            return
        except Exception as e:
            # Fallback: intentar ffmpeg si está disponible
            if shutil.which("ffmpeg"):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        "pipe:0",
                        "-f",
                        "wav",
                        "pipe:1",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    out, err = await proc.communicate(audio_bytes)

                    if proc.returncode != 0:
                        raise RuntimeError(
                            "ffmpeg failed to decode audio: "
                            + err.decode(errors="ignore")
                        )

                    bio2 = io.BytesIO(out)
                    data, samplerate = sf.read(bio2, dtype="float32")
                    sd.play(data, samplerate)
                    sd.wait()
                    print("🔊 TTS reproducido en memoria (ffmpeg fallback)")
                    return
                except Exception as ffmpeg_err:
                    raise RuntimeError(
                        f"Failed to decode audio with ffmpeg: {ffmpeg_err}"
                    )

            raise RuntimeError(
                "No se pudo decodificar el audio en memoria. "
                "Instala ffmpeg o habilita soporte MP3 en libsndfile. "
                f"Original error: {e}"
            )

    async def synthesize_bytes_async(self, text: str) -> bytes:
        """
        Genera audio desde texto y devuelve los bytes generados.

        - Si USE_KOKORO está activo y disponible -> WAV bytes de Kokoro.
        - Si no -> bytes de edge-tts (generalmente MP3/OGG).
        - Si USE_RVC está activo y disponible -> los bytes devueltos vienen
          pasados por RVC (voz personalizada).
        """
        if not text:
            return b""

        if self.use_kokoro and self._kokoro_available:
            audio_bytes = await self._synthesize_kokoro_bytes_async(text)
            if not audio_bytes:
                audio_bytes = await self._synthesize_edge_bytes_async(text)
        else:
            audio_bytes = await self._synthesize_edge_bytes_async(text)

        if not audio_bytes:
            return b""

        # Paso intermedio: aplicar RVC si procede (con fallback)
        audio_bytes = self._maybe_apply_rvc(audio_bytes)

        return audio_bytes

    def synthesize_bytes(self, text: str) -> bytes:
        """
        Versión sincrónica de `synthesize_bytes_async`.
        """
        return asyncio.run(self.synthesize_bytes_async(text))

    def speak(self, text: str):
        """
        Genera audio desde texto y lo reproduce (versión síncrona).
        """
        asyncio.run(self.speak_async(text))


def run_tts_thread(text: str, tts: TTSEngine):
    """Feedback verbal rápido en hilo aparte."""
    def _speak():
        tts.speak(text)
    thread = threading.Thread(target=_speak)
    thread.start()