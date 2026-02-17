"use client";

import { useState } from "react";

function EyeIcon() {
	return (
		<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
			<circle cx="12" cy="12" r="3" />
		</svg>
	);
}

function EyeOffIcon() {
	return (
		<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
			<path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
			<line x1="1" y1="1" x2="23" y2="23" />
			<path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
		</svg>
	);
}

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
	label: string;
	error?: string;
}

export default function Input({ label, error, type = "text", id, ...rest }: InputProps) {
	const [showPassword, setShowPassword] = useState(false);
	const isPassword = type === "password";

	return (
		<div className="flex flex-col gap-1.5">
			<label htmlFor={id} className="text-sm text-white/60">
				{label}
			</label>
			<div className="relative">
				<input
					id={id}
					type={isPassword && showPassword ? "text" : type}
					className={`w-full bg-white/5 border rounded-xl px-4 py-3 text-white placeholder:text-white/30 outline-none transition-colors ${error ? "border-red-500 focus:border-red-500 focus:ring-1 focus:ring-red-500/30" : "border-white/10 focus:border-green-500 focus:ring-1 focus:ring-green-500/30"}`}
					{...rest}
				/>
				{isPassword && (
					<button type="button" onClick={() => setShowPassword((prev) => !prev)} className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/70 transition-colors" aria-label={showPassword ? "Hide password" : "Show password"}>
						{showPassword ? <EyeOffIcon /> : <EyeIcon />}
					</button>
				)}
			</div>
			{error && <p className="text-sm text-red-400">{error}</p>}
		</div>
	);
}
