import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders LLM Performance Monitor heading', () => {
  render(<App />);
  const headingElement = screen.getByText(/LLM Performance Monitor/i);
  expect(headingElement).toBeInTheDocument();
});
