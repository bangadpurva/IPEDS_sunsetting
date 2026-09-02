import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Pathwise — Evidence-backed education and career pathways',
  description: 'Explore degrees, careers, and colleges using transparent IPEDS and BLS evidence.',
  openGraph: {
    title: 'Pathwise — Choose a path with evidence, not guesswork.',
    description: 'Explore degrees, careers, and colleges using transparent IPEDS and BLS evidence.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Pathwise student pathway explorer' }],
  },
  twitter: { card: 'summary_large_image', images: ['/og.png'] },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
