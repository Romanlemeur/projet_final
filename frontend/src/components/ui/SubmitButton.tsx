interface SubmitButtonProps {
	children: React.ReactNode;
}

export default function SubmitButton({ children }: SubmitButtonProps) {
	return (
		<button type="submit" className="w-full bg-green-500 hover:bg-green-600 text-black font-bold py-3 rounded-xl transition-colors mt-2 cursor-pointer">
			{children}
		</button>
	);
}
