import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function LoginForm() {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate(); // For navigation after login

    const handleSubmit = async (event) => {
      event.preventDefault();

      try {
        // Use the proxy for the API endpoint
        const response = await axios.post("/api/api/v1/users/login", {
          email,
          username,
          password,
        });

        if (response.status === 200) {
          // If login is successful, redirect to /home
          navigate("/");
        }
      } catch (error) {
        console.error("Login failed:", error.response?.data || error.message);
        alert("Invalid credentials. Please try again.");
      }
    };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <label htmlFor="email">Email: </label>
        <br />
        <input
          type="email"
          name="email"
          id="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)} // Update state
          required
        />
        <br />
        <label htmlFor="username">Username: </label>
        <br />
        <input
          type="text"
          name="username"
          id="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)} // Update state
          required
        />
        <br />
        <label htmlFor="password">Password: </label>
        <br />
        <input
          type="password"
          name="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)} // Update state
          required
        />
        <br />
        <input type="submit" value="Login" />
      </form>
    </div>
  );
}
