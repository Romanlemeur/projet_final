"use client";

import { useFormStatus } from "react-dom";

interface SubmitButtonProps {
	children: React.ReactNode;
}

export default function SubmitButton({ children }: SubmitButtonProps) {
	const { pending } = useFormStatus();

	return (
		<button type="submit" disabled={pending} className="w-full bg-green-500 hover:bg-green-600 disabled:opacity-60 disabled:cursor-not-allowed text-black font-bold py-3 rounded-xl transition-colors mt-2 cursor-pointer flex items-center justify-center gap-2">
			{pending && (
				<svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
					<circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
					<path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
				</svg>
			)}
			{children}
		</button>
	);
}
