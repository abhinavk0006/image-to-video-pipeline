"""Generate clip videos locally from an input image and motion prompt.

This module intentionally does not embed Wan 2.2 specifics. Instead, it wraps a
configurable local command so the pipeline can run inside Lightning AI while the
underlying generation script remains swappable.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from models import PromptBundle


class VideoGenerationError(RuntimeError):
	"""Raised when a local video generation command fails."""


def is_cuda_oom_error(message: str) -> bool:
	"""Return whether a generator failure is a CUDA out-of-memory failure."""
	normalized = message.lower()
	return (
		"out of memory" in normalized
		or "cuda_error_out_of_memory" in normalized
		or "cublas_status_alloc_failed" in normalized
	)


@dataclass(slots=True)
class VideoGenerationRequest:
	"""Inputs needed to generate a single clip video."""

	input_image_path: str | Path
	output_video_path: str | Path
	prompt_bundle: PromptBundle
	clip_name: str = ""
	metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LocalVideoGenerator:
	"""Runs a configurable local generation command and validates its output."""

	executable: str = "python"
	script_path: str | None = None
	base_arguments: list[str] = field(default_factory=list)
	working_directory: str | Path | None = None
	max_area: int | None = None
	fallback_areas: tuple[int, ...] = ()
	_process: subprocess.Popen | None = field(default=None, init=False, repr=False)
	_use_daemon: bool = field(default=True, init=False)
	_stderr_lines: list[str] = field(default_factory=list, init=False, repr=False)
	_stderr_thread: threading.Thread | None = field(default=None, init=False, repr=False)

	def generate_clip_video(self, request: VideoGenerationRequest) -> Path:
		"""Run the configured local command to produce one clip video."""

		input_image_path = Path(request.input_image_path).resolve()
		output_video_path = Path(request.output_video_path).resolve()

		if not input_image_path.exists():
			raise VideoGenerationError(f"Input image does not exist: {input_image_path}")

		output_video_path.parent.mkdir(parents=True, exist_ok=True)

		areas = self._generation_areas()

		# Try daemon mode if enabled
		if self._use_daemon:
			if self._process is None:
				# Start the daemon process
				command = [self.executable, self.script_path, *self.base_arguments, "--daemon"]
				try:
					self._stderr_lines = []
					self._process = subprocess.Popen(
						command,
						stdin=subprocess.PIPE,
						stdout=subprocess.PIPE,
						stderr=subprocess.PIPE,
						text=True,
						cwd=str(self.working_directory) if self.working_directory is not None else None,
					)
					self._stderr_thread = threading.Thread(
						target=self._drain_stderr,
						args=(self._process.stderr,),
						daemon=True,
					)
					self._stderr_thread.start()
					
					# Read READY without blocking the pipeline forever. Wan initialization can
					# legitimately take several minutes, so this is a heartbeat interval rather
					# than a hard startup timeout.
					ready_line = self._read_daemon_stdout("startup")
					if ready_line.strip() != "READY":
						self._use_daemon = False
						if self._process:
							self._process.kill()
							self._process = None
					else:
						# Daemon started successfully!
						pass
				except Exception:
					self._use_daemon = False
					if self._process:
						self._process.kill()
						self._process = None

			if self._use_daemon and self._process is not None:
				import json
				motion_prompt = request.prompt_bundle.motion_prompt or self._compose_motion_prompt(request.prompt_bundle)
				for attempt, area in enumerate(areas):
					if self._process is None:
						self._use_daemon = True
						# The failed daemon can only be replaced after an OOM-qualified
						# termination; the response path below decides that.
						command = [self.executable, self.script_path, *self.base_arguments, "--daemon"]
						self._stderr_lines = []
						self._process = subprocess.Popen(
							command,
							stdin=subprocess.PIPE,
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							text=True,
							cwd=str(self.working_directory) if self.working_directory is not None else None,
						)
						self._stderr_thread = threading.Thread(
							target=self._drain_stderr,
							args=(self._process.stderr,),
							daemon=True,
						)
						self._stderr_thread.start()
						if self._read_daemon_stdout("restart").strip() != "READY":
							raise VideoGenerationError(
								"Video generation daemon failed to restart."
							)
					task = {
						"input_image": str(input_image_path),
						"output_video": str(output_video_path),
						"prompt": motion_prompt,
						"clip_duration": request.prompt_bundle.clip_duration_seconds,
						"negative_prompt": request.prompt_bundle.negative_prompt,
						"clip_name": request.clip_name,
					}
					if area is not None:
						task["max_area"] = area

					try:
						self._process.stdin.write(json.dumps(task) + "\n")
						self._process.stdin.flush()

						response_line = self._read_daemon_stdout(
							f"clip {request.clip_name or output_video_path.name} at area "
							f"{area if area is not None else 'default'}"
						)
						if not response_line:
							diagnostics = self._daemon_diagnostics()
							was_oom = is_cuda_oom_error(diagnostics)
							exit_code = self._process.poll()
							self._process = None
							if was_oom and attempt + 1 < len(areas):
								print(
									f"CUDA OOM daemon termination at area "
									f"{area if area is not None else 'default'} "
									f"(exit code {exit_code}); retrying at area "
									f"{areas[attempt + 1]}. Diagnostics: {diagnostics}",
									file=sys.stderr,
								)
								output_video_path.unlink(missing_ok=True)
								continue
							raise VideoGenerationError(
								f"Video generation daemon terminated at area "
								f"{area if area is not None else 'default'} "
								f"(exit code {exit_code}). Diagnostics: "
								f"{diagnostics or 'none'}"
							)

						response = json.loads(response_line)
						if response.get("status") == "success":
							if not output_video_path.exists():
								raise VideoGenerationError(
									"Video generation completed but no file was written: "
									f"{output_video_path}"
								)
							print(
								f"Generated {request.clip_name or output_video_path.name} "
								f"at area {area if area is not None else 'default'}.",
								file=sys.stderr,
							)
							return output_video_path

						error_message = str(response.get("error", "unknown daemon error"))
						error_message = (
							f"{error_message}; daemon stderr: "
							f"{self._daemon_diagnostics()}"
						)
						if is_cuda_oom_error(error_message) and attempt + 1 < len(areas):
							print(
								f"CUDA OOM at area {area if area is not None else 'default'}; "
								f"retrying at area {areas[attempt + 1]}.",
								file=sys.stderr,
							)
							output_video_path.unlink(missing_ok=True)
							continue
						raise VideoGenerationError(
							f"Video generation daemon error at area "
							f"{area if area is not None else 'default'}: {error_message}"
						)
					except VideoGenerationError:
						if self._process is not None:
							try:
								self._process.kill()
							except Exception:
								pass
							self._process = None
						raise
					except Exception as error:
						if self._process is not None:
							try:
								self._process.kill()
							except Exception:
								pass
							self._process = None
						raise VideoGenerationError(
							f"Daemon generation failed at area "
							f"{area if area is not None else 'default'}: {error}"
						) from error
				raise AssertionError("generation area fallback loop was empty")

		# Fallback to standard subprocess execution
		for attempt, area in enumerate(areas):
			command = self._build_command(
				request, input_image_path, output_video_path, max_area=area
			)
			try:
				subprocess.run(
					command,
					check=True,
					capture_output=True,
					text=True,
					cwd=str(self.working_directory) if self.working_directory is not None else None,
				)
			except FileNotFoundError as error:
				raise VideoGenerationError(
					f"Could not start video generation command: {command[0]}"
				) from error
			except subprocess.CalledProcessError as error:
				details = (error.stderr or error.stdout or "").strip()
				if is_cuda_oom_error(details) and attempt + 1 < len(areas):
					print(
						f"CUDA OOM at area {area if area is not None else 'default'}; "
						f"retrying at area {areas[attempt + 1]}.",
						file=sys.stderr,
					)
					output_video_path.unlink(missing_ok=True)
					continue
				raise VideoGenerationError(
					f"Video generation failed at area "
					f"{area if area is not None else 'default'} for "
					f"{request.clip_name or output_video_path.name}: {details}"
				) from error

			if not output_video_path.exists():
				raise VideoGenerationError(
					"Video generation completed but no file was written: "
					f"{output_video_path} (selected area "
					f"{area if area is not None else 'default'})"
				)
			print(
				f"Generated {request.clip_name or output_video_path.name} at area "
				f"{area if area is not None else 'default'}.",
				file=sys.stderr,
			)
			return output_video_path

		raise AssertionError("generation area fallback loop was empty")

	def _drain_stderr(self, stream) -> None:
		"""Continuously capture daemon diagnostics without blocking stdout replies."""
		if stream is None:
			return
		for line in stream:
			line = line.rstrip()
			self._stderr_lines.append(line)
			# Wan's useful progress/diagnostics are intentionally on stderr because
			# stdout is reserved for the JSON-lines control protocol. Forward them
			# live so Kaggle users can see that the persistent worker is alive.
			if line:
				print(f"[wan-daemon] {line}", file=sys.stderr, flush=True)

	def _read_daemon_stdout(self, phase: str) -> str:
		"""Read one protocol line while emitting periodic liveness heartbeats."""
		if self._process is None or self._process.stdout is None:
			raise VideoGenerationError("Video generation daemon has no stdout pipe")

		result: queue.Queue[str] = queue.Queue(maxsize=1)

		def reader() -> None:
			try:
				line = self._process.stdout.readline()
				result.put(line)
			except Exception as error:
				result.put("")
				print(
					f"[wan-daemon] stdout reader failed during {phase}: {error}",
					file=sys.stderr,
					flush=True,
				)

		thread = threading.Thread(target=reader, daemon=True)
		thread.start()
		while True:
			try:
				return result.get(timeout=30.0)
			except queue.Empty:
				process = self._process
				exit_code = process.poll() if process is not None else None
				diagnostics = self._daemon_diagnostics()
				print(
					f"[wan-daemon] still waiting for {phase} response "
					f"(process exit={exit_code if exit_code is not None else 'running'}). "
					f"Recent diagnostics: {diagnostics[-1000:] if diagnostics else 'none'}",
					file=sys.stderr,
					flush=True,
				)

	def _daemon_diagnostics(self) -> str:
		if self._stderr_thread is not None:
			self._stderr_thread.join(timeout=1)
		return "\n".join(line for line in self._stderr_lines if line).strip()

	def stop_daemon(self) -> None:
		"""Exit the daemon process cleanly."""
		if self._process is not None:
			import json
			try:
				self._process.stdin.write(json.dumps({"action": "exit"}) + "\n")
				self._process.stdin.flush()
				self._process.wait(timeout=5)
			except Exception:
				try:
					self._process.kill()
				except Exception:
					pass
			self._process = None

	def _build_command(
		self,
		request: VideoGenerationRequest,
		input_image_path: Path,
		output_video_path: Path,
		*,
		max_area: int | None = None,
	) -> list[str]:
		if self.script_path is None:
			raise VideoGenerationError("script_path must be set to run local video generation")

		motion_prompt = request.prompt_bundle.motion_prompt or self._compose_motion_prompt(request.prompt_bundle)

		command = [self.executable, self.script_path, *self.base_arguments]
		command.extend([
			"--input-image",
			str(input_image_path),
			"--output-video",
			str(output_video_path),
			"--prompt",
			motion_prompt,
			"--clip-duration",
			str(request.prompt_bundle.clip_duration_seconds),
		])

		if request.prompt_bundle.negative_prompt:
			command.extend(["--negative-prompt", request.prompt_bundle.negative_prompt])

		if request.clip_name:
			command.extend(["--clip-name", request.clip_name])
		if max_area is not None:
			command.extend(["--max-area", str(max_area)])

		for key, value in request.metadata.items():
			command.extend([f"--{key.replace('_', '-')}", value])

		return command

	def _generation_areas(self) -> tuple[int | None, ...]:
		areas = (self.max_area, *self.fallback_areas)
		if not areas:
			return (None,)
		if any(area is not None and area <= 0 for area in areas):
			raise VideoGenerationError("Configured generation areas must be greater than zero")
		if len(set(areas)) != len(areas):
			raise VideoGenerationError("Configured generation areas must be unique")
		if any(
			areas[index] is not None
			and areas[index + 1] is not None
			and areas[index + 1] >= areas[index]
			for index in range(len(areas) - 1)
		):
			raise VideoGenerationError("Fallback generation areas must strictly decrease")
		return areas

	@staticmethod
	def _compose_motion_prompt(prompt_bundle: PromptBundle) -> str:
		parts = [
			prompt_bundle.motion,
			f"moving object: {prompt_bundle.moving_object}" if prompt_bundle.moving_object else "",
			f"direction: {prompt_bundle.direction}" if prompt_bundle.direction else "",
			f"constraints: {prompt_bundle.constraints}" if prompt_bundle.constraints else "",
			f"stop condition: {prompt_bundle.stop_condition}" if prompt_bundle.stop_condition else "",
		]
		return ", ".join(part for part in parts if part)


def generate_clip_video(
	input_image_path: str | Path,
	output_video_path: str | Path,
	prompt_bundle: PromptBundle,
	*,
	clip_name: str = "",
	executable: str = "python",
	script_path: str | None = None,
	base_arguments: Sequence[str] | None = None,
	working_directory: str | Path | None = None,
	metadata: dict[str, str] | None = None,
) -> Path:
	"""Convenience wrapper for the default local generator."""

	generator = LocalVideoGenerator(
		executable=executable,
		script_path=script_path,
		base_arguments=list(base_arguments or []),
		working_directory=working_directory,
	)
	request = VideoGenerationRequest(
		input_image_path=input_image_path,
		output_video_path=output_video_path,
		prompt_bundle=prompt_bundle,
		clip_name=clip_name,
		metadata=dict(metadata or {}),
	)
	return generator.generate_clip_video(request)
