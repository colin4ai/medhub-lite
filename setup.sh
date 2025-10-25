#!/bin/bash
# Setup script for MedHub Lite

echo "🏥 Setting up MedHub Lite..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your OpenAI API key!"
    echo "   Get your key from: https://platform.openai.com/api-keys"
    echo ""
fi

# Create necessary directories
mkdir -p data/chroma_db
mkdir -p data/sample_documents
mkdir -p data/evaluation

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the CLI: python cli.py"
echo "4. Or run the API server: python api.py"
echo ""
echo "Quick test:"
echo "  python cli.py"
echo "  medhub> add data/sample_documents/sample_medical_record.txt"
echo "  medhub> ask What are the patient's work restrictions?"
echo ""
