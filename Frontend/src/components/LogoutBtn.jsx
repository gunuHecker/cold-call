import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

export default function LogoutBtn() {
  const navigate = useNavigate();

  const handleClick = async () => {
    try {
      const response = await axios.post("/api/api/v1/users/logout");

      if (response.status === 200) {
        navigate("/home");
      }
    } catch (error) {
      console.error("Logout failed:", error.response?.data || error.message);
      alert("Logout Failed. Please try again.");
    }
  };

  return (
    <div>
      <button onClick={handleClick}>Logout</button>
    </div>
  );
}
