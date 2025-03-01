import { useState } from 'react'
import React from "react";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import HomePage from "./pages/HomePage"
import LoginPage from "./pages/LoginPage"
import RegisterPage from "./pages/RegisterPage"
import AudioPage from "./pages/AudioPage"
import './App.css'


function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: (
        <HomePage />
      ),
    },
    {
      path: "/users/register",
      element: (
        <RegisterPage />
      ),
    },
    {
      path: "/users/login",
      element: (
        <LoginPage />
      ),
    },
    {
      path: "/audio",
      element: (
        <AudioPage />
      ),
    },
  ]);

  return (
    <>
      <RouterProvider router={router} />
    </>
  );
}

export default App
