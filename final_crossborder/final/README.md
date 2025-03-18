# Union Bank Cross-Border Transaction System

A full-stack web application for handling international transactions with multi-language support and transaction history tracking.

## Features
- Secure user authentication
- Cross-border money transfers
- Multi-language support
- Transaction history dashboard
- Excel export functionality
- Real-time currency conversion

## Tech Stack
- Frontend: React.js with TypeScript
- Backend: Node.js with Express
- Database: MongoDB
- Authentication: JWT
- Styling: Tailwind CSS

## Project Structure
```
union-bank/
├── client/           # Frontend React application
├── server/           # Backend Node.js application
└── README.md
```

## Setup Instructions

### Prerequisites
- Node.js (v14 or higher)
- MongoDB
- npm or yarn

### Frontend Setup
```bash
cd client
npm install
npm start
```

### Backend Setup
```bash
cd server
npm install
npm start
```

## Environment Variables
Create .env files in both client and server directories:

### Client (.env)
```
REACT_APP_API_URL=http://localhost:5000
```

### Server (.env)
```
PORT=5000
MONGODB_URI=your_mongodb_uri
JWT_SECRET=your_jwt_secret
```

## API Documentation
Detailed API documentation can be found in the server/docs directory. 