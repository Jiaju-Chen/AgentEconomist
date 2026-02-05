/**
 * API Route to serve experiment files (YAML configs, etc.)
 * 
 * This route reads files from the AgentEconomist project directory
 * and serves them to the frontend for display (e.g., config diff)
 */

import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const filePath = searchParams.get('path');
    
    if (!filePath) {
      return NextResponse.json(
        { error: 'Missing path parameter' },
        { status: 400 }
      );
    }
    
    // Security: Only allow reading from experiment_files directory
    if (!filePath.startsWith('/experiment_files/') && !filePath.startsWith('experiment_files/')) {
      return NextResponse.json(
        { error: 'Access denied: Only experiment_files are accessible' },
        { status: 403 }
      );
    }
    
    // Construct absolute path
    // Assuming the project root is 2 levels up from frontend/src/app/api/files
    const projectRoot = join(process.cwd(), '..');
    const absolutePath = join(projectRoot, filePath.replace(/^\//, ''));
    
    console.log('[Files API] Reading file:', absolutePath);
    
    // Read file
    const content = await readFile(absolutePath, 'utf-8');
    
    return new NextResponse(content, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'no-cache',
      },
    });
  } catch (error) {
    console.error('[Files API] Error:', error);
    
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return NextResponse.json(
        { error: 'File not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
