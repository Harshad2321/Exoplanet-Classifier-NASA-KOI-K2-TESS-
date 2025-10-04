import React, { useRef, useEffect } from 'react';

interface TextPressureProps {
  text: string;
  className?: string;
}

export const TextPressure: React.FC<TextPressureProps> = ({ text, className }) => {
  const containerRef = useRef<HTMLHeadingElement>(null);
  const lettersRef = useRef<Array<HTMLSpanElement | null>>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      lettersRef.current.forEach(span => {
        if (!span) return;

        const spanRect = span.getBoundingClientRect();
        const spanCenterX = (spanRect.left - rect.left) + spanRect.width / 2;
        const spanCenterY = (spanRect.top - rect.top) + spanRect.height / 2;

        const deltaX = mouseX - spanCenterX;
        const deltaY = mouseY - spanCenterY;

        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        
        const maxDistance = 100;

        if (distance < maxDistance) {
          const scale = 1 + (1 - distance / maxDistance) * 0.5;
          const opacity = 0.5 + (1 - distance / maxDistance) * 0.5; 
          const translateY = (1 - distance / maxDistance) * -10;

          span.style.transform = `scale(${scale}) translateY(${translateY}px)`;
          span.style.opacity = `${opacity}`;
          span.style.zIndex = '1';
        } else {
          span.style.transform = 'scale(1) translateY(0px)';
          span.style.opacity = '1';
          span.style.zIndex = '0';
        }
      });
    };

    const handleMouseLeave = () => {
        lettersRef.current.forEach(span => {
            if (!span) return;
            span.style.transform = 'scale(1) translateY(0px)';
            span.style.opacity = '1';
            span.style.zIndex = '0';
        });
    };

    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  return (
    <h1 ref={containerRef} className={className} aria-label={text}>
      {text.split('').map((char, index) => (
        <span
          key={index}
          // Fix: The ref callback should not return a value. An assignment expression returns the
          // assigned value, so `(lettersRef.current[index] = el)` was returning the element.
          // Using a block body `{...}` ensures the function implicitly returns undefined.
          ref={el => { lettersRef.current[index] = el; }}
          style={{
            display: 'inline-block',
            transition: 'transform 0.1s ease-out, opacity 0.1s ease-out',
            userSelect: 'none'
          }}
          aria-hidden="true"
        >
          {char === ' ' ? '\u00A0' : char}
        </span>
      ))}
    </h1>
  );
};
